# Code Structure

## Overview

RoVer-CoRe verifies controller safety under state estimation uncertainty using Hamilton-Jacobi reachability. The codebase separates abstract interfaces (`src/core/`) from concrete implementations (`src/impl/`).

## Core Abstractions

### System ([src/core/systems.py](../src/core/systems.py))

Defines a dynamical system with state, control, disturbance, and uncertainty.

```python
class System:
    # Dimensions
    state_dim: int
    control_dim: int
    disturbance_dim: int
    
    # Dynamics
    def dynamics(self, state, control, disturbance, time) -> Tensor
    
    # Bounds
    def control_limits(self, state, time) -> Tuple[Tensor, Tensor]
    def disturbance_limits(self, state, time) -> Tuple[Tensor, Tensor]
    def uncertainty_limits(self, state, time) -> Tuple[Tensor, Tensor]
    
    # Objectives
    def failure_function(self, state, time) -> Tensor  # < 0 means failure
    def goal_function(self, state, time) -> Tensor     # < 0 means goal reached
```

**Implementations:** `RoverDark`, `RoverLight`

---

### Input ([src/core/inputs.py](../src/core/inputs.py))

Maps (state, time) to an action vector. Used for control, disturbance, or uncertainty.

```python
class Input:
    type: Literal['control', 'disturbance', 'uncertainty', 'any']
    system_class: Type[System]
    dim: int
    time_invariant: bool
    
    def input(self, state, time) -> Tensor
    def bind(self, system) -> None
```

**Implementations:**
- Standalone: `ZeroInput`, `UniformRandomInput`, `RoverDark_MPC`
- Derived: `GridInput`, `NNInput`, `OptimalInputFromValue`

---

### Set ([src/core/sets.py](../src/core/sets.py))

Represents a set of possible inputs at each (state, time). Captures how state uncertainty affects control.

```python
class Set:
    set_type: Literal['box', 'hull']
    dim: int
    
    def as_box(self, state, time) -> Tuple[Tensor, Tensor]
    def argmax_support(self, direction, state, time) -> Tensor
```

**Implementation:** `GridSet` (grid-cached boxes or convex hulls)

---

### Value ([src/core/values.py](../src/core/values.py))

Value function V(x,t) from HJ reachability. Encodes backward reachable tube.

```python
class Value:
    def value(self, state, time) -> Tensor
    def gradient(self, state, time) -> Tensor
```

**Implementation:** `GridValue` (grid-cached values and gradients)

---

### HJReachabilityDynamics ([src/core/hj_reachability.py](../src/core/hj_reachability.py))

Configures HJ solver with control/disturbance/uncertainty channels.

```python
class HJReachabilityDynamics:
    system: Type[System]
    control: ChannelConfig      # GIVEN (set), OPTIMIZE, or ZERO
    disturbance: ChannelConfig
    uncertainty: ChannelConfig
    
    def bind_control_set(self, set: Set) -> None
    def optimal_uncertainty_from_grad(self, state, time, grad) -> Tensor
```

**Implementations:** (in `src/impl/hj_reachability/`)
- `RoverDark` — Robust BRT with worst-case uncertainty (control as Set)
- `RoverDarkNominal` — Nominal BRT with deterministic control (control as Input)

> **Note:** Use `--failure-grid-value-tag` to compute a BRT using a precomputed
> nominal BRT as the failure function (for recursive verification).

---

## Directory Layout

```
src/
├── core/                    # Abstract interfaces
│   ├── systems.py           # System base class
│   ├── inputs.py            # Input base class
│   ├── sets.py              # Set base class
│   ├── values.py            # Value base class
│   ├── hj_reachability.py   # HJ dynamics configuration
│   └── simulators.py        # Euler and discrete simulators
│
├── impl/                    # Concrete implementations
│   ├── systems/             # RoverDark, RoverLight, etc.
│   ├── inputs/              # MPC controllers, derived inputs
│   ├── sets/                # GridSet, JaxGridSet
│   ├── values/              # GridValue
│   └── hj_reachability/     # HJ dynamics implementations
│
└── utils/                   # Shared utilities
    ├── registry.py          # Class discovery and instantiation
    ├── cache_loaders.py     # Load cached GridInput/Set/Value
    ├── config.py            # YAML configuration loading
    ├── grids.py             # Grid indexing utilities
    └── obstacles.py         # Obstacle primitives (Circle2D, Box2D)
```

---

## Configuration Files

| File | Purpose |
|------|---------|
| `config/simulations.yaml` | Default simulation parameters per system |
| `config/resolutions.yaml` | Grid resolutions for caches |
| `config/nn_training.yaml` | Neural network training settings |
| `config/visualizations.yaml` | Visualization slice configurations |
