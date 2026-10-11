"""Caller-owned synchronous host extension delegating to the existing local worker.

This example observes the actual local interpreter and direct-worker wait deadline.
It provides no memory/network/process-tree/container isolation or provider support.
"""


def local(context, *, operation):
    if operation == "inspect":
        observed = context.inspect_local()
        return {
            "available": True,
            "environment": observed["observed"],
            "capabilities": observed["backend"]["capabilities"],
            "container_verification": "not_run",
            "host": {"name": "caller-delegated-local-worker"},
            "scope": "actual local interpreter; direct worker only",
            "installs_dependencies": False,
        }
    if operation == "run":
        return context.run_local()
    raise ValueError("unknown runtime backend operation")


def unavailable(context, *, operation):
    if operation != "inspect":
        raise AssertionError("an unavailable backend must never execute application code")
    return {
        "available": False,
        "capabilities": [],
        "reason": "caller-selected host is unavailable",
        "container_verification": "not_run",
        "installs_dependencies": False,
    }


def independent_local(context, *, operation, interpreter):
    """A caller selects an actual target separately from a controller-only path."""
    from dataclasses import replace

    return local(replace(context, python=interpreter), operation=operation)


def fail_after_marker(context, *, operation):
    if operation == "inspect":
        return local(context, operation=operation)
    context.outputs["marker"].write_text("one attempted host effect", encoding="utf-8")
    raise RuntimeError("caller backend failed after its attempted effect")
