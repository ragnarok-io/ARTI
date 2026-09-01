# Adaptive Observation

`AdaptiveObservation` builds a bounded observation trajectory from one immutable
tensor substrate. Every observation reads the original input; observations are
not recursively fed into later observations.

```python
import arti

policy = arti.mechanisms.LearnedObservationPolicy(
    input_dim=64,
    state_dim=16,
    max_observations=8,
    min_observations=1,
)
operator = arti.mechanisms.StateAffineObservationOperator(
    dim=64,
    state_dim=16,
)
observation = arti.mechanisms.AdaptiveObservation(
    policy,
    operator=operator,
    executor="early_break",
)
```

The input is a `WORLD` `TensorEnvelope`. The result has shape
`[B, T_max, N, D]` and carries a boolean observation mask. Individual samples
may use different counts while the physical output shape remains bounded and
batch friendly.

## Components

- `FixedObservationPolicy` supplies a fixed trajectory and remains the reference
  policy for deterministic layouts.
- `LearnedObservationPolicy` derives recurrent observation states and a
  continuation decision from the masked input tensor.
- `IdentityObservationOperator` repeats the unchanged substrate.
- `StateAffineObservationOperator` applies a bounded, state-conditioned feature
  frame while preserving shape.
- `FourierShiftObservationOperator` applies a differentiable circular subpixel
  translation to a declared 2D tensor layout. It accepts Cartesian displacement
  or polar radius plus a normalized direction vector.
- `BankConditionedObservationPolicy` uses fixed Query coordinates and typed
  observation operands to form a bounded trajectory from one or more Banks.
- `AdaptiveObservation` executes the policy and operator under tensor and memory
  limits.

The policy and operator are independent modules. A custom policy may control
trajectory length without defining observation mathematics; a custom operator
may implement another tensor observation without owning stopping behavior.

## Fourier and polar observation

`FourierShiftObservationOperator` treats `[B, H*W, D]` as an `H x W` field and
applies the Fourier shift theorem. The operator always observes the original
substrate and uses circular boundaries; it does not recursively resample an
earlier observation.

```python
operator = arti.mechanisms.FourierShiftObservationOperator(
    spatial_shape=(32, 32),
    state_mode="polar",
    compile_policy="safe_training",
)
```

Cartesian state is `[dx, dy]`. Polar state is `[radius, direction_x,
direction_y]`; the direction pair is normalized by the operator, and a zero
direction produces zero displacement. `safe_training` is the default: an outer
`torch.compile` graph is allowed to break at the native `torch.fft` boundary,
so forward and backward remain reliable without maintaining a second Fourier
implementation. `fullgraph_forward` opts into a full compiled inference graph;
fullgraph complex-FFT backward is not claimed by the current stable PyTorch
release. `execution_contract()` reports these capabilities explicitly.

## Bank-conditioned trajectories

```python
bank = arti.mechanisms.ObservationOperandBank(
    slots=32,
    key_dim=16,
    factor_dim=4,  # three state factors plus continuation
    bank_id="camera-path",
)
policy = arti.mechanisms.BankConditionedObservationPolicy(
    input_dim=64,
    state_dim=3,
    max_observations=8,
    banks=[bank],
)
observation = arti.mechanisms.AdaptiveObservation(policy, operator=operator)
```

The Query is fixed and deterministic. Each Bank normalizes its own routes; an
explicit `bank_weights` vector combines multiple Banks without silently making
their slots compete in one global softmax. The Bank supplies typed observation
operands, the Formula interprets state and continuation factors, and the
operator owns the actual observation mathematics. These roles remain
replaceable and independently versioned.

## Learned stopping

The learned policy returns a hard boolean mask and a same-shaped activity
weight. Its forward value is exactly zero or one. Its backward path uses a
straight-through estimator so an ordinary downstream loss can train the
continuation controller. There is no separate training/evaluation stopping
rule: identical inputs and parameters produce identical masks in both modes.

`static_masked` executes the configured maximum number of operator calls and
masks inactive observations. `early_break` skips calls after every sample in
the batch has stopped, while retaining the same bounded output shape. During
`torch.compile`, execution remains static and masked to preserve a full graph.

This API does not imply that more observations improve every task. Task quality,
active observation count, and runtime cost must be measured separately.
