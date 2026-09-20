# Concepts

ARTI is organized around a few composable mechanisms:

- **Coordinates** describe the observation frame of the input tensor.
- **Frame inverse** can transform observed tensors back into a canonical latent frame.
- **Mask and visibility** control which tokens exist and which tokens may influence each other.
- **Virtual interface** provides internal latent work surfaces for structured token synchronization.
- **Virtual recall** aligns corrupted-input auxiliary outputs to clean-input latent targets during training.
- **Diagnostics** expose internal weights, gates, coverage, and residual norms.

The strongest current ARTI identity is the combination:

```text
coord frame inverse
  -> visibility-constrained virtual interface
  -> dynamic latent update
  -> virtual recall alignment
```
