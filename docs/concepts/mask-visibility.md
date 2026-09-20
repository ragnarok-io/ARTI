# Mask And Visibility

Mask and visibility are first-class ARTI inputs.

`mask` controls token validity:

```text
mask: [B, N]
```

Invalid tokens should not contribute to pooling, routing, synchronization, or diagnostics statistics.

`visibility` controls token-to-token influence:

```text
visibility: [B, N, N]
```

Visibility answers a different question from mask:

```text
valid token != visible token
visible token != authorized influence
```

ARTI uses visibility in pairwise context and virtual interface synchronization.
