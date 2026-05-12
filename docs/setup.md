# Setup

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended for JAX HJ solver)

## Installation

```bash
# Create virtual environment
python -m venv env
source env/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install HJ reachability solver (editable)
pip install -e libraries/hj_reachability

# Install auto_LiRPA for NN verification (editable)
# Use --no-deps to avoid numpy version conflict with jax
pip install --no-deps -e libraries/auto_LiRPA
```

## Dependencies

**Core:**
- `torch` — Neural networks, tensor operations, simulation
- `jax` / `jaxlib` — HJ reachability solver (GPU-accelerated PDE solving)
- `hj_reachability` — Hamilton-Jacobi toolbox (local library)

**Controllers:**
- `casadi` / `do_mpc` — Model predictive control

**Verification:**
- `auto_LiRPA` — Neural network bound propagation (local library)

**Visualization:**
- `matplotlib` — Plotting and animation
- `wandb` — Experiment tracking (optional)

## Directory Setup

The following directories are created automatically:

```
.cache/
├── grid_inputs/    # Cached control evaluations
├── grid_sets/      # Cached control sets under uncertainty
├── grid_values/    # Cached HJ value functions
└── nn_inputs/      # Trained neural network surrogates

outputs/
└── simulations/    # Simulation results and visualizations
```

## Verification

Test that imports work:

```bash
python -c "from src.impl.systems.rover_dark import RoverDark; print('OK')"
```

List available systems and inputs:

```bash
python scripts/simulation/simulate.py --list
```
