"""Bounded device-side finite predicates over independent tensor requirements."""

import torch


class _FiniteTensorRows:
    def __init__(self):
        self.tensors = []
        self.identities = {}
        self.objects = {}
        self.requirements = []

    def add(self, values):
        indices = []
        identities, tensors = self.identities, self.tensors
        for value in values:
            index = self.objects.get(id(value))
            if index is not None:
                indices.append(index)
                continue
            # Match the addressed region, never merely its storage/base Tensor.
            key = (value.data_ptr(), tuple(value.shape), value.stride(), value.dtype, value.device)
            index = identities.get(key)
            if index is None:
                index = len(tensors) + 1
                identities[key] = index
                tensors.append(value)
                # Only memoize objects retained above, so ids cannot be reused.
                self.objects[id(value)] = index
            indices.append(index)
        self.requirements.append(indices)

    def evaluate(self, *, device):
        if not self.requirements:
            return torch.ones(0, dtype=torch.bool, device=device)
        flags = torch.ones(len(self.tensors) + 1, dtype=torch.bool, device=device)
        groups = {}
        for index, value in enumerate(self.tensors, 1):
            groups.setdefault((tuple(value.shape), value.dtype, value.device), []).append((index, value))
        for group in groups.values():
            chunk_size = max(1, 262144 // max(1, group[0][1].numel()))
            for start in range(0, len(group), chunk_size):
                chunk = group[start:start + chunk_size]
                numeric = torch.stack([value.detach() for _, value in chunk])
                valid = torch.isfinite(numeric).reshape(len(chunk), -1).all(dim=1)
                flags[torch.tensor([index for index, _ in chunk], device=device)] = valid.to(device)
        arity = max(1, max(map(len, self.requirements)))
        indices = torch.tensor([row + [0] * (arity - len(row)) for row in self.requirements], device=device)
        return flags[indices].all(dim=1)
