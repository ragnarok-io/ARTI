# Virtual Interface

The virtual interface is ARTI's internal latent work surface.

It is not long-term memory and not external recall. It is a fixed-size set of learned latent desktops where token fragments can be placed, synchronized, and read back.

```text
token latent fragments
  -> virtual interface slots
  -> visibility-constrained latent context
  -> token update
```

## Why It Exists

Full token-to-token interaction grows as `O(N^2)`. The virtual interface provides an `O(N*S)` communication path where `S` is the number of interface slots.

The deeper reason is architectural: ARTI should not require every token to directly influence every other token. Information can pass through a limited set of internal work surfaces, which gives the model a stronger structural prior.

## Visibility

When visibility is provided, the interface respects token-to-token influence constraints. A token can only read interface state written from visible source tokens.

This makes the interface compatible with padding masks, causal masks, authority gates, and other tensor-level visibility constraints.
