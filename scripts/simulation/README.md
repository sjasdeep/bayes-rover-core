# Simulation Scripts

This folder contains scripts for simulating, inspecting, and visualizing closed-loop system trajectories.

## Workflow

1. **simulate.py** - Run closed-loop simulations and save results to pickle files
2. **inspect_simulation.py** - Inspect saved simulation data and metadata
3. **visualize_simulation.py** - Generate videos and frames from saved simulations

## simulate.py
Forward-Euler simulation script. Saves simulation results as pickle files with tensor data.

```bash
# List available systems and compatible inputs
python scripts/simulation/simulate.py --list

# Run a simulation with config-driven defaults
python scripts/simulation/simulate.py \
    --system RoverDark \
    --tag my_simulation

# Run with specific initial state
python scripts/simulation/simulate.py \
    --system RoverDark \
    --tag my_simulation \
    --initial-state 0.0 0.0 0.0

# Run with preset initial states (batch simulation)
python scripts/simulation/simulate.py \
    --system RoverDark \
    --tag my_simulation \
    --initial-states-preset default_batch

# Use cache-backed inputs for control/disturbance/uncertainty
python scripts/simulation/simulate.py \
    --system RoverDark \
    --tag my_simulation \
    --control OptimalInputFromValue \
    --control-tag my_control_cache_tag \
    --disturbance GridInput \
    --disturbance-tag my_disturbance_cache_tag \
    --uncertainty NNInput \
    --uncertainty-tag my_uncertainty_cache_tag
```

**Note:** Any of `--control`, `--disturbance`, or `--uncertainty` can be `OptimalInputFromValue`, `GridInput`, or `NNInput`. Provide the cache `--{input-type}-tag`.

**Outputs:** `outputs/simulations/{TAG}/`
- `results.pkl` - Simulation trajectories (tensors) and metadata
- `metadata.json` - Lightweight metadata for fast listing

## inspect_simulation.py
Inspect saved simulation results.

```bash
# List all saved simulations
python scripts/simulation/inspect_simulation.py --list

# Inspect a specific simulation
python scripts/simulation/inspect_simulation.py --tag my_simulation

# Inspect with verbose output (show all details)
python scripts/simulation/inspect_simulation.py --tag my_simulation --verbose
```

## visualize_simulation.py
Generate videos and frames from saved simulation results.

```bash
# Generate video and frames from saved simulation
python scripts/simulation/visualize_simulation.py --tag my_simulation

# Generate with custom FPS
python scripts/simulation/visualize_simulation.py --tag my_simulation --fps 30

# Generate frames only (no video)
python scripts/simulation/visualize_simulation.py --tag my_simulation --no-video

# Force regeneration even if output exists
python scripts/simulation/visualize_simulation.py --tag my_simulation --force
```

**Outputs:** Added to `outputs/simulations/{TAG}/`
- `frames/` - Individual PNG frames
- `simulation.mp4` - Rendered video (unless --no-video)