# GridInput Scripts

Utilities for building, inspecting, and visualizing GridInput caches.

## build_grid_input.py
Build GridInput cache for a specific system-input pair.<br>
Uses grids from [config/resolutions.yaml](../../config/resolutions.yaml).

```bash
# List available systems and inputs
python scripts/grid_input/build_grid_input.py --list

# Build cache for a specific combination
python scripts/grid_input/build_grid_input.py \
    --system RoverDark \
    --input RoverDark_MPC \
    --tag {TAG} \
    --description {DESCRIPTION}
```

**Output:** [.cache/grid_inputs/{TAG}.pkl](../../.cache/grid_inputs/)

## inspect_grid_input_cache.py
List all cached grids and their info (tag, description, system, input, grid shape, disk usage).

```bash
python scripts/grid_input/inspect_grid_input_cache.py
```

## visualize_grid_input.py
Visualize cached GridInput in 2D slices.<br>
Uses presets from [config/visualizations.yaml](../../config/visualizations.yaml).

```bash
# Static mode (default) - saves images to disk
python scripts/grid_input/visualize_grid_input.py \
    --tag {TAG} \
    --preset {PRESET} \     # optional
    --save-dir {SAVE_DIR} \ # optional
    --interpolate           # optional

# Interactive mode - explore with sliders
python scripts/grid_input/visualize_grid_input.py \
    --tag {TAG} \
    --preset {PRESET} \     # optional
    --interpolate \         # optional
    --interactive
```

**Output:** [outputs/visualizations/grid_inputs/{TAG}/{PRESET}/](../../outputs/visualizations/grid_inputs/)

**Interactive mode** opens a window with sliders to adjust time and fixed dimensions in real-time.