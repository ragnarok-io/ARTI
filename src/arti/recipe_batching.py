"""Bounded cross-request batching outside model execution."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
import math
from typing import Any, Callable


@dataclass(frozen=True)
class RecipeServingConfig:
    batching: str = "auto"
    max_batch_size: int = 32
    batch_wait_ms: float = 4.0
    max_pending_records: int = 1024

    def __post_init__(self) -> None:
        if self.batching not in {"auto", "on", "off"}:
            raise ValueError("serving.batching must be auto, on or off")
        for name in ("max_batch_size", "max_pending_records"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"serving.{name} must be a positive integer")
        if (
            isinstance(self.batch_wait_ms, bool)
            or not isinstance(self.batch_wait_ms, (int, float))
            or not math.isfinite(self.batch_wait_ms)
            or self.batch_wait_ms < 0
        ):
            raise ValueError("serving.batch_wait_ms must be finite and nonnegative")


class _ServingUnavailable(RuntimeError):
    pass


@dataclass
class _Request:
    records: list[dict[str, Any]]
    kwargs: dict[str, Any]
    future: asyncio.Future
    submitted: float
    offset: int = 0
    results: list[Any] = field(default_factory=list)
    released: bool = False


class _BatchDispatcher:
    def __init__(self, predict: Callable, config: RecipeServingConfig, *, dynamic: bool) -> None:
        self.predict, self.config, self.dynamic = predict, config, dynamic
        self._groups: dict[Any, deque[_Request]] = {}
        self._wake: asyncio.Event | None = None
        self._task: asyncio.Task | None = None
        self._active: list[_Request] = []
        self.accepting = True
        self._sequence = 0
        self.pending_records = 0
        self.stats = {
            "accepted_requests": 0,
            "completed_requests": 0,
            "cancelled_requests": 0,
            "failed_requests": 0,
            "rejected_requests": 0,
            "predictor_calls": 0,
            "processed_records": 0,
            "largest_batch": 0,
            "model_seconds": 0.0,
            "queue_wait_seconds": 0.0,
        }

    def start(self) -> None:
        if not self.accepting:
            raise _ServingUnavailable("inference worker is closing")
        if self._task is None:
            self._wake = asyncio.Event()
            self._task = asyncio.create_task(self._run(), name="arti-recipe-batches")

    async def submit(self, records, **kwargs):
        if not self.accepting:
            raise _ServingUnavailable("inference worker is closing")
        if not records:
            return []
        if self.pending_records + len(records) > self.config.max_pending_records:
            self.stats["rejected_requests"] += 1
            raise _ServingUnavailable("inference queue is full; retry later")
        self.start()
        loop = asyncio.get_running_loop()
        request = _Request(list(records), dict(kwargs), loop.create_future(), loop.time())
        self._sequence += 1
        key = tuple(sorted(kwargs.items())) if self.dynamic else self._sequence
        self._groups.setdefault(key, deque()).append(request)
        self.pending_records += len(records)
        self.stats["accepted_requests"] += 1
        self._wake.set()
        try:
            return await request.future
        except asyncio.CancelledError:
            self.stats["cancelled_requests"] += 1
            self._release(request)
            self._wake.set()
            raise

    def _release(self, request: _Request) -> None:
        if not request.released:
            self.pending_records -= len(request.records)
            request.released = True

    def _fail(self, request: _Request, error: Exception) -> None:
        if not request.future.done():
            request.future.set_exception(error)
            self.stats["failed_requests"] += 1
        self._release(request)

    async def _run(self) -> None:
        try:
            while self.accepting or self._groups:
                if not self._groups:
                    self._wake.clear()
                    await self._wake.wait()
                    continue
                key = next(iter(self._groups))
                group = self._groups[key]
                while group and group[0].future.done():
                    self._release(group.popleft())
                if not group:
                    self._groups.pop(key, None)
                    continue
                limit = (
                    min(
                        self.config.max_batch_size,
                        group[0].kwargs.get("batch_size", self.config.max_batch_size),
                    )
                    if self.dynamic
                    else len(group[0].records)
                )
                deadline = group[0].submitted + self.config.batch_wait_ms / 1000
                while (
                    self.dynamic
                    and self.accepting
                    and sum(
                        len(item.records) - item.offset for item in group if not item.future.done()
                    )
                    < limit
                ):
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        break
                    self._wake.clear()
                    try:
                        await asyncio.wait_for(self._wake.wait(), remaining)
                    except asyncio.TimeoutError:
                        break
                records, pieces = [], []
                while group and len(records) < limit:
                    request = group[0]
                    if request.future.done():
                        self._release(group.popleft())
                        continue
                    count = min(limit - len(records), len(request.records) - request.offset)
                    records.extend(request.records[request.offset : request.offset + count])
                    request.offset += count
                    pieces.append((request, count))
                    if request.offset == len(request.records):
                        group.popleft()
                if group:
                    self._groups[key] = self._groups.pop(key)
                else:
                    self._groups.pop(key, None)
                if not records:
                    continue
                self._active = [request for request, _ in pieces]
                now = asyncio.get_running_loop().time()
                self.stats["queue_wait_seconds"] += sum(
                    now - request.submitted for request in self._active
                )
                self.stats["predictor_calls"] += 1
                self.stats["largest_batch"] = max(self.stats["largest_batch"], len(records))
                try:
                    values = await asyncio.to_thread(self.predict, records, **pieces[0][0].kwargs)
                    if len(values) != len(records):
                        raise RuntimeError("predictor returned a different number of records")
                    offset = 0
                    for request, count in pieces:
                        if not request.future.done():
                            request.results.extend(values[offset : offset + count])
                            if len(request.results) == len(request.records):
                                request.future.set_result(request.results)
                                self.stats["completed_requests"] += 1
                                self._release(request)
                        offset += count
                    self.stats["processed_records"] += len(records)
                except asyncio.CancelledError:
                    for request in self._active:
                        self._fail(request, _ServingUnavailable("inference worker cancelled"))
                    raise
                except Exception as error:
                    for request in self._active:
                        self._fail(request, error)
                finally:
                    self.stats["model_seconds"] += asyncio.get_running_loop().time() - now
                    self._active = []
        finally:
            self.accepting = False
            error = _ServingUnavailable("inference worker closed")
            for group in self._groups.values():
                for request in group:
                    self._fail(request, error)
            self._groups.clear()
            for request in self._active:
                self._fail(request, error)

    async def close(self) -> None:
        self.accepting = False
        for group in self._groups.values():
            for request in group:
                self._fail(request, _ServingUnavailable("inference worker is closing"))
        self._groups.clear()
        if self._wake is not None:
            self._wake.set()
        if self._task is not None:
            await asyncio.shield(self._task)

    def snapshot(self) -> dict[str, Any]:
        return {
            **self.stats,
            "pending_records": self.pending_records,
            "dynamic_batching": self.dynamic,
            "accepting": self.accepting,
        }


__all__ = ["RecipeServingConfig"]
