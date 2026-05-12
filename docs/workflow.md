# Workflow

The verification pipeline has four stages, each producing a cached artifact.

## Pipeline Overview

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  GridInput  │ -> │   GridSet   │ -> │  GridValue  │ -> │ Simulation  │
│  (control)  │    │ (ctrl sets) │    │   (BRT)     │    │ (verify)    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

---

## Stage 1: Build GridInput

**Purpose:** Cache control outputs on a state-time grid.

```bash
python scripts/grid_input/build_grid_input.py \
    --system RoverDark \
    --input RoverDark_MPC \
    --tag RoverDark_MPC \
    --description "MPC control on 100³ grid"
```

**Output:** `.cache/grid_inputs/{TAG}.pkl`

**Inspect:** `python scripts/grid_input/inspect_grid_input_cache.py --tag {TAG}`

---

## Stage 2: Build GridSet

**Purpose:** Compute control sets capturing uncertainty-induced variability.

```bash
python scripts/grid_set/build_grid_set.py \
    --system RoverDark \
    --grid-input-tag RoverDark_MPC \
    --tag RoverDark_MPC \
    --set-type hull \
    --description "Convex hull control sets"
```

**Output:** `.cache/grid_sets/{TAG}.pkl`

**Options:**
- `--set-type box` — Axis-aligned bounding box (faster, more conservative)
- `--set-type hull` — Convex hull (tighter, slower)

---

## Stage 3: Build GridValue

**Purpose:** Solve HJ PDE to compute backward reachable tube.

```bash
python scripts/grid_value/build_grid_value.py \
    --dynamics RoverDark \
    --control-grid-set-tag RoverDark_MPC \
    --tag RoverDark_WorstCase \
    --description "BRT under worst-case uncertainty"
```

**Output:** `.cache/grid_values/{TAG}.pkl`

**Inspect:** `python scripts/grid_value/inspect_grid_value_cache.py --tag {TAG}`

**Visualize:**
```bash
python scripts/grid_value/visualize_grid_value.py --tag RoverDark_WorstCase
```

---

## Stage 4: Simulate and Verify

**Purpose:** Run closed-loop simulations under worst-case uncertainty.

```bash
python scripts/simulation/simulate.py \
    --system RoverDark \
    --control GridInput \
    --control-tag RoverDark_MPC \
    --uncertainty OptimalInputFromValue \
    --uncertainty-tag RoverDark_WorstCase \
    --dt 0.01 \
    --tag RoverDark_Verification
```

**Output:** `outputs/simulations/{TAG}/`

**Visualize:**
```bash
python scripts/simulation/visualize_simulation.py --tag RoverDark_Verification
```

**Options:**
- `--preset {name}` — Use preset from `config/simulations.yaml`

---

## Optional: Neural Network Surrogate

Train a neural network to approximate a GridInput for faster evaluation:

```bash
python scripts/nn_input/train_nn_input.py \
    --system RoverDark \
    --input-class GridInput \
    --input-tag RoverDark_MPC \
    --tag RoverDark_MPC_NN
```

Then compute NN-based control sets using auto_LiRPA bounds:

```bash
python scripts/nn_input/build_grid_set.py \
    --system RoverDark \
    --nn-input-tag RoverDark_MPC_NN \
    --tag RoverDark_MPC_NN_Set
```

---

## Configuration

All scripts read defaults from `config/` YAML files:

- `simulations.yaml` — Simulation parameters, initial state presets
- `resolutions.yaml` — Grid resolutions per system
- `nn_training.yaml` — Training hyperparameters, W&B logging

CLI arguments override config file values.

---

## Listing Available Resources

```bash
# List systems, inputs, and cached artifacts
python scripts/simulation/simulate.py --list

# List GridValue caches
python scripts/grid_value/inspect_grid_value_cache.py

# List GridSet caches  
python scripts/grid_set/inspect_grid_set_cache.py

# List GridInput caches
python scripts/grid_input/inspect_grid_input_cache.py
```

