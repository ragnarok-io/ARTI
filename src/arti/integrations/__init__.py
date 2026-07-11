"""Optional integrations for external model families.

Integrations keep heavyweight third-party dependencies behind explicit imports.
Install the matching optional extra before using them, for example
``uv sync --extra qwen`` for Qwen/Transformers helpers.
"""

from .qwen import GlyphRuntimeReadout, QwenDialogueDrift, QwenGlyphRuntimeAdapter, QwenGlyphRuntimeConfig

__all__ = [
    "GlyphRuntimeReadout",
    "QwenDialogueDrift",
    "QwenGlyphRuntimeAdapter",
    "QwenGlyphRuntimeConfig",
]
