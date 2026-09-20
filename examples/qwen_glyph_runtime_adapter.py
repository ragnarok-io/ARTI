"""Use ARTI's alpha Qwen glyph runtime adapter.

Install optional dependencies first:

    uv sync --extra torch --extra qwen --extra font

The adapter keeps ordinary Qwen generation on the frozen base model path and
uses a separate ARTI glyph runtime-vocab readout for external visible words.
"""

from __future__ import annotations

from arti.integrations.qwen import QwenGlyphRuntimeAdapter


def main() -> None:
    adapter = QwenGlyphRuntimeAdapter.from_pretrained("Qwen/Qwen3-0.6B")
    prompt = "User: Read the external visible word.\nAssistant:"
    words = ["strawberry", "strawberrry", "banana", "phase"]

    dialogue = adapter.generate("User: Say hello briefly.\nAssistant:", max_new_tokens=16)
    readout = adapter.read_glyph_vocab(prompt, words, query_text="strawberrry")
    drift = adapter.dialogue_drift(
        ["User: Hello, who are you?\nAssistant:", "User: What is 2 plus 3?\nAssistant:"],
        operation=lambda: adapter.read_glyph_vocab(prompt, words, query_text="phase"),
    )

    print("dialogue:", dialogue)
    print("glyph_local_index:", readout.local_index)
    print("glyph_text:", readout.text)
    print("dialogue_max_abs_logit_delta:", drift.max_abs_logit_delta)


if __name__ == "__main__":
    main()
