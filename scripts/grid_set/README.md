# GridSet Scripts

Utilities for building, inspecting, and visualizing GridSet caches.

## build_grid_set.py
Build GridSet cache for a specific system and GridInput cache pair, with chosen set type.<br>
Reuses the exact state grid from the referenced GridInput cache. For time:
- If the GridInput cache is time-varying, we reuse its time grid.
- If the GridInput cache is time-invariant but the system's uncertainty limits vary with time, we synthesize a time grid using the system's `time_horizon` and a time resolution from `config/resolutions.yaml`.

```bash
# List available systems and GridInput caches (by tag)
python scripts/grid_set/build_grid_set.py --list

# Build cache for a specific combination
python scripts/grid_set/build_grid_set.py \
    --system RoverDark \
    --grid-input-tag {GRID_INPUT_TAG} \
    --tag {TAG} \
    --description {DESCRIPTION} \
    --set-type hull \
    --config config/resolutions.yaml \   # optional, used only when GI is time-invariant
    --time-resolution 100                # optional override
```

**Output:** [.cache/grid_sets/{TAG}.pkl](../../.cache/grid_sets/)

## inspect_grid_set_cache.py
List all cached grid sets and their info (tag, description, system, input, set type, grid shape, disk usage). No flags; just prints a table and notes any incomplete files.

```bash
python scripts/grid_set/inspect_grid_set_cache.py
```

## visualize_grid_set.py
Visualize cached GridSet in 2D slices.<br>
Uses presets from [config/visualizations.yaml](../../config/visualizations.yaml).

### Static Mode (default)
Generates and saves PNG files for all slices defined in the preset.

```bash
python scripts/grid_set/visualize_grid_set.py \
    --tag {TAG} \
    --preset {PRESET} \     # optional
    --save-dir {SAVE_DIR} \ # optional
    --interpolate           # optional
```

**Output:** [outputs/visualizations/grid_sets/{TAG}/{PRESET}/](../../outputs/visualizations/grid_sets/)

### Interactive Mode
Launch an interactive window with sliders to explore the GridSet bounds dynamically.

```bash
python scripts/grid_set/visualize_grid_set.py \
    --tag {TAG} \
    --preset {PRESET} \     # optional
    --interpolate \         # optional
    --interactive
```

**Features:**
- Time slider (if time-variant)
- Fixed dimension sliders (for dimensions not being plotted)
- Input dimension selector (to choose which input dimension to visualize)
- Real-time updates showing lower bound, upper bound, uncertainty width, and nominal/midpoint

Notes:
- Supported set types: box and hull.
- Hull visualizations use the box approximation (min/max per grid point). With --interpolate, we interpolate the box bounds.
- Cache management mirrors GridInput: manual tag management, no overwrite (errors if tag exists), atomic writes, and clear diagnostics for empty/truncated cache files.
- Time handling: A time-invariant GridInput can still yield a time-varying GridSet when the system’s uncertainty limits vary with time. In that case, the builder uses System.time_horizon and a time resolution from config to evaluate bounds across time.