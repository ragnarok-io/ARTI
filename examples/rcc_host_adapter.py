"""Capture C after several host layers and reuse it at the native input head."""

import torch
from torch import nn

from arti.alpha import RCCHostAdapter, RecursiveContextCompiler


class InputHead(nn.Module):
    def __init__(self, dim: int, max_length: int) -> None:
        super().__init__()
        self.positions = nn.Embedding(max_length, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.shape[1], device=x.device)
        return x + self.positions(positions)[None]


class Host(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tokens = nn.Embedding(64, 16)
        self.input_head = InputHead(16, 128)
        self.early = nn.TransformerEncoderLayer(16, 4, batch_first=True)
        self.late = nn.TransformerEncoderLayer(16, 4, batch_first=True)
        self.language_head = nn.Linear(16, 64)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        length = token_ids.shape[1]
        x = self.input_head(self.tokens(token_ids))
        x = self.early(x)
        x = self.late(x)
        return self.language_head(x[:, -length:])


def bind_at_input(context, args, kwargs):
    # InputHead applies this host's learned positions to the combined stream.
    return (torch.cat((context, args[0]), dim=1),), kwargs


torch.manual_seed(0)
adapter = RCCHostAdapter(
    Host(),
    RecursiveContextCompiler(16, context_width=4, heads=4, layers=2),
    capture_paths=("early", "late"),
    consume_path="input_head",
    bind_context=bind_at_input,
)
first = adapter(torch.tensor([[1, 2, 3]]))
second = adapter(torch.tensor([[4, 5]]), context=first.context)
print(first.output.shape, first.context.shape, second.output.shape)
