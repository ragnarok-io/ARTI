"""Minimal RCC forward: recursive C is internal; only normal logits leave the model."""

import torch

from arti.alpha import (
    ContextEvaluator,
    LanguageHead,
    LexicalEmbedding,
    RecursiveContextCompiler,
    RecursiveLanguageModel,
    SinusoidalTokenPositionEncoder,
)
from arti.recursive_language import prefix_plan


torch.manual_seed(0)
model = RecursiveLanguageModel(
    ContextEvaluator(
        RecursiveContextCompiler(32, context_width=8, heads=4, layers=3, readout_layers=(1, 3)),
        LexicalEmbedding(
            256,
            32,
            position_encoder=SinusoidalTokenPositionEncoder(32, base=10000.0),
        ),
    ),
    LanguageHead(32, 256),
).eval()
plan = prefix_plan(
    range(24),
    [24],
    chunk_size=4,
    depth=2,
    fanout=2,
    context_width=4,
    compile_depth=3,
)
with torch.no_grad():
    logits = model(plan)
print(logits.shape)
