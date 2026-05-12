# RoVer-CoRe

**Ro**bust **Ver**ification of **Co**ntrollers via Hamilton-Jacobi (HJ) **Re**achability.

[[Paper]](https://arxiv.org/abs/2511.14755v1)

This codebase implements RoVer-CoRe, which aims to verify the safety of state-based controllers operating under uncertainty by using HJ reachability analysis to account for worst-case estimation errors.

## Quick Start

### Setup

```bash
python -m venv env && source env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e libraries/hj_reachability
pip install --no-deps -e libraries/auto_LiRPA  # NN verification (Example C)
```

### Core Pipeline (required for all examples)

```bash
# Stage 1: Build GridInput (MPC control evaluated on state grid) [~7 minutes]
python scripts/grid_input/build_grid_input.py --system RoverDark --input RoverDark_MPC --tag RoverDark_MPC

# Stage 2: Build GridSet (control sets capturing uncertainty) [~30 minutes]
python scripts/grid_set/build_grid_set.py --system RoverDark --grid-input-tag RoverDark_MPC --tag RoverDark_MPC_Box

# Stage 3: Build GridValue (BRT for worst-case estimation error) [<1 minute]
python scripts/grid_value/build_grid_value.py --dynamics RoverDark --control-grid-set-tag RoverDark_MPC_Box --tag RoverDark_WorstCase
```

### Example A: RoverDark (worst-case uncertainty in the dark)

```bash
# Simulate batch of challenging initial states under worst-case uncertainty [<1 minute]
python scripts/simulation/simulate.py --system RoverDark --preset default --tag roverdark_sim
python scripts/simulation/visualize_simulation.py --tag roverdark_sim
```

<details>
<summary><b>Example B: RoverLight (safety-preserving light policy)</b></summary>

```bash
# Build nominal BRT (no uncertainty - uses GridInput with RoverDarkNominal dynamics) [<1 minute]
python scripts/grid_value/build_grid_value.py --dynamics RoverDarkNominal --control-grid-input-tag RoverDark_MPC --tag RoverDark_Nominal

# Build recursive BRT (uses nominal BRT as failure function) [<1 minute]
python scripts/grid_value/build_grid_value.py --dynamics RoverDark --preset recursive_brt --control-grid-set-tag RoverDark_MPC_Box --tag RoverDark_Recursive

# Simulate light policy under worst-case uncertainty [<1 minute]
python scripts/simulation/simulate.py --system RoverLight --tag roverlight_batch
python scripts/simulation/visualize_simulation.py --tag roverlight_batch --save-final-frame --same-color
```

</details>

<details>
<summary><b>Example C: Neural network controller</b></summary>

```bash
# Train NN to approximate MPC (from GridInput cache) [~1.5 hours]
python scripts/nn_input/train_nn_input.py --system RoverDark --input-tag RoverDark_MPC --tag RoverDark_MPC_NN --hidden 128 --layers 2 --epochs 500000

# Compute NN output bounds via auto_LiRPA, then clamp to control limits [~1.5 hours]
python scripts/nn_input/build_grid_set.py --system RoverDark --nn-input-tag RoverDark_MPC_NN --tag RoverDark_MPC_NN_Box --method CROWN-IBP
python scripts/grid_set/constrain_grid_set.py --grid-set-tag RoverDark_MPC_NN_Box --tag RoverDark_MPC_NN_Box_Clamped

# Build recursive BRT (uses nominal BRT as failure function) [<1 minute]
python scripts/grid_value/build_grid_value.py --dynamics RoverDark --preset recursive_brt --control-grid-set-tag RoverDark_MPC_NN_Box_Clamped --tag RoverDark_Recursive_NN

# Simulate light policy under worst-case uncertainty [<1 minute]
python scripts/simulation/simulate.py --system RoverLight --set control_grid_value_tag=RoverDark_Recursive_NN --tag roverlight_batch_nn
python scripts/simulation/visualize_simulation.py --tag roverlight_batch_nn --save-final-frame --same-color
```

</details>

## Documentation

| Document | Description |
|----------|-------------|
| [Setup](docs/setup.md) | Environment setup and dependencies |
| [Structure](docs/structure.md) | Code architecture and core abstractions |
| [Workflow](docs/workflow.md) | Step-by-step verification pipeline |

## Project Structure

```
RoVer-CoRe/
├── src/
│   ├── core/          # Abstract interfaces (System, Input, Set, Value)
│   ├── impl/          # Concrete implementations
│   └── utils/         # Shared utilities
├── scripts/           # Workflow scripts (build, simulate, visualize)
├── config/            # YAML configuration files
├── .cache/            # Intermediate computation results
└── outputs/           # Simulation outputs and visualizations
```

## Key Concepts

- **System**: Dynamical system with state, control, disturbance, and uncertainty channels
- **Input**: Maps (state, time) → action for control, disturbance, or uncertainty
- **Set**: Represents the range of possible control actions under state uncertainty
- **Value**: HJ value function V(x,t) encoding the backward reachable tube
- **GridInput/GridSet/GridValue**: Grid-based caches for efficient computation

## Example Systems

| System | State | Description |
|--------|-------|-------------|
| `RoverDark` | (x, y, θ) | Dubins car with configurable uncertainty growth |
| `RoverLight` | (x, y, θ, s) | Dubins car with controllable "light" that resets uncertainty |

## Citation

```bibtex
@article{lin2025robust,
  title={Robust Verification of Controllers under State Uncertainty via Hamilton-Jacobi Reachability Analysis},
  author={Lin, Albert and Pinto, Alessandro and Bansal, Somil},
  journal={arXiv preprint arXiv:2511.14755},
  year={2025}
}
```
