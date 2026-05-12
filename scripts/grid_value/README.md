# GridValue Scripts

Utilities for building, inspecting, and visualizing GridValue caches.

## build_grid_value.py

Build GridValue cache for a specific `hj_reachability.Dynamics` in [src/impl/hj_reachability](../../src/impl/hj_reachability/).<br>
May require supplying a control, disturbance, and/or uncertainty.<br>
Uses grids from [config/resolutions.yaml](../../config/resolutions.yaml).

```bash
# List available hj_reachability.Dynamics and inputs
python scripts/grid_value/build_grid_value.py --list

# Build cache for a specific combination
python scripts/grid_value/build_grid_value.py \
    --dynamics RoverDark \
    --control-grid-set-tag {GRID_SET_TAG} \
    # other inputs, depending on --dynamics
    # --(con./dis./unc.)-input {Input}
    # --(con./dis./unc.)-grid-input-tag {GRID_INPUT_TAG} \
    --tag {TAG} \
    --description {DESCRIPTION}
```

**Output:** [.cache/grid_values/{TAG}.pkl](../../.cache/grid_values/)

---

## inspect_grid_value_cache.py

List all cached grids and their info (tag, description, dynamics, inputs, grid shape, disk usage).

```bash
python scripts/grid_value/inspect_grid_value_cache.py
```

---

## visualize_grid_value.py

Visualize cached GridValue in 2D slices.<br>
Uses presets from [config/visualizations.yaml](../../config/visualizations.yaml).

### Static Mode (default)
Generates and saves PNG files for specified time slices and evolution plots.

```bash
python scripts/grid_value/visualize_grid_value.py \
    --tag {TAG} \
    --preset {PRESET} \     # optional
    --save-dir {SAVE_DIR} \ # optional
    --interpolate           # optional
```

**Output:** [outputs/visualizations/grid_values/{TAG}/{PRESET}/](../../outputs/visualizations/grid_values/)

### Interactive Mode
Launch an interactive window with sliders to explore the backward reachable tube dynamically.

```bash
python scripts/grid_value/visualize_grid_value.py \
    --tag {TAG} \
    --interpolate \         # optional
    --interactive \
    --slice-dim 2 \         # optional (for 3D+ state spaces)
    --slice-value 0.0       # optional (initial slice value)
```

**Features:**
- Time slider to scrub through backward reachable tube evolution
- Slice value slider (for 3D+ state spaces) to explore different slices
- Real-time updates showing value function heatmap and zero-level set boundary
- Obstacle visualization (if system has obstacles)


**Output:** [outputs/visualizations/grid_values/{TAG}/{PRESET}/](../../outputs/visualizations/grid_values/)