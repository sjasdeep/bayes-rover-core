# Value Evaluation

This folder contains utilities to compare value-based reachable-set estimates against baselines, Monte Carlo under-approximations, and simulation results.

## compare_values.py

Overlay the primary GridValue with optional baselines, MC value, and simulation overlays. Produces a single comparison figure or an interactive window.

### Key overlays

- Primary GridValue
  - Filled contour of the value (symmetric colormap centered at 0)
  - Zero level set V=0 (solid black)
- Baseline GridValue (optional)
  - Zero level set V=0 (dashed black)
- Monte Carlo value (optional)
  - Final snapshot only, zero-level set (dashed black)
- Simulation overlays
  - If `--sim-tag` is a grid-based simulation: overlay zero-level set of the per-trajectory minimum failure (magenta dash-dot)
  - Otherwise: plot trajectories clipped at first failure (blue), with start marker (hollow blue circle) and collision marker (red “x”)
  - `--sim-tag2` lets you add trajectories from a second (typically non-grid) simulation tag; uses the same styling

### Basic usage

- Primary vs baseline:

```bash
python scripts/value_evaluation/compare_values.py \
  --value-tag RoverDark_WorstCase \
  --baseline-tag RoverDark_WorstCase_no_uncertainty
```

- Add MC final zero-level and trajectories from a non-grid simulation:

```bash
python scripts/value_evaluation/compare_values.py \
  --value-tag RoverDark_WorstCase \
  --mc-tag RoverDark_MCValue \
  --sim-tag RoverDark_Simulation
```

- Grid simulation overlay (min-failure zero-level) + extra non-grid trajectories:

```bash
python scripts/value_evaluation/compare_values.py \
  --value-tag RoverDark_WorstCase \
  --sim-tag RoverDark_Simulation_FullGrid \
  --sim-tag2 RoverDark_Simulation_Sample
```

- Interactive (no files or directories are created):

```bash
python scripts/value_evaluation/compare_values.py \
  --value-tag RoverDark_WorstCase \
  --sim-tag RoverDark_Simulation \
  --interactive
```

### Helpful options

- `--time 0.0` Set the time at which to slice the GridValue (default: 0.0)
- `--slice-dim 2` and `--slice-value v` Choose which dimension to hold fixed for 3D+ values
- `--xlim xmin xmax` and `--ylim ymin ymax` Zoom the plot to a specific window
- `--save-dir DIR` Write the figure to a custom directory; otherwise defaults to
  `outputs/visualizations/value_evaluation/val_{VALUE}__[...combination...]`

### Styling notes

- Legend is placed outside the plot on the right to avoid occlusion
- Lines are slightly thinned for readability:
  - Robust V=0: solid black
  - Baseline V=0: dashed black
  - MC final V=0: dashed black
  - Min-failure V=0 (grid sim): magenta dash-dot
  - Trajectories: blue (estimated trajectories shown as dashed blue)
  - Start markers: small hollow blue circles; collision markers: red “x”

