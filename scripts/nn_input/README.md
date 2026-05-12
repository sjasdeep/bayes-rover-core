# NNInput Scripts

Train and save neural network approximations of Inputs (NNInput caches).

## train_nn_input.py
Train a PyTorch MLP to approximate an Input(state[, time]).
Reads defaults from [config/nn_training.yaml](../../config/nn_training.yaml).

Outputs: [.cache/nn_inputs/{TAG}.pth + .meta.json](../../.cache/nn_inputs/)

### Examples

Train from a cached GridInput (tag)

```bash
python scripts/nn_input/train_nn_input.py \
  --system RoverDark \
  --input-class GridInput \
  --input-tag {GRID_INPUT_TAG} \
  --type control \
  --tag {NN_TAG}
# Note: --num-samples is optional for GridInput; if omitted, uses ALL grid points
```

Train from a value function (OptimalInputFromValue) using a GridValue tag

```bash
python scripts/nn_input/train_nn_input.py \
  --system RoverDark \
  --input-class OptimalInputFromValue \
  --input-tag {GRID_VALUE_TAG} \
  --type control \
  --num-samples 20000 \
  --tag {NN_TAG}
```

Train from a direct Input implementation (e.g., ZeroInput)

```bash
python scripts/nn_input/train_nn_input.py \
  --system RoverDark \
  --input-class ZeroInput \
  --type control \
  --num-samples 10000 \
  --tag {NN_TAG}
```

Use config presets from nn_training.yaml

```yaml
# config/nn_training.yaml
MySystem:
  default:
    source:
      input_class: GridInput          # or OptimalInputFromValue, NNInput, ZeroInput, ...
      input_tag: my_gridinput_tag     # required for cache-backed classes
      type: control
    num_samples: 50000                # omit for GridInput to use all points
    model: {hidden: 128, layers: 3}
    train: {epochs: 20, batch_size: 1024, lr: 0.001}
    tag: my_nninput_tag
```

```bash
# With the above defaults
python scripts/nn_input/train_nn_input.py --system MySystem
# Or select a preset under MySystem.default.presets
python scripts/nn_input/train_nn_input.py --system MySystem --preset quick
```

### Required vs optional arguments

- Required:
  - --system
  - source.type (via --type or config)
  - model.hidden, model.layers (via CLI or config)
  - train.batch_size, train.lr, train.epochs (via CLI or config)
  - Tag to save under (via --tag or config tag)
  - For cache-backed inputs (GridInput, OptimalInputFromValue, NNInput): source.input_tag or --input-tag
  - For non-GridInput sources: --num-samples or config num_samples
- Optional:
  - For GridInput sources, num_samples may be omitted to use the full grid
  - --device (defaults to CUDA if available, else CPU)

### Weights & Biases logging (optional)

Enable via CLI or config (must have wandb installed):

```bash
python scripts/nn_input/train_nn_input.py \
  --system RoverDark \
  --input-class GridInput --input-tag {GRID_INPUT_TAG} --type control \
  --tag {NN_TAG} \
  --wandb --wandb-project my-project [--wandb-entity my-team] [--wandb-mode online] [--wandb-tags a b]
```

Or in config:

```yaml
logging:
  wandb:
    enable: true
    project: my-project
    # entity: my-team
    # mode: online  # or offline
    # tags: [nninput, debug]
```

The run logs the training MSE per epoch and captures config metadata (system, input type/class/tag, model sizes, dataset size, etc.).

## inspect_nn_input_cache.py
List cached NNInput models (tags) with system, type, sizes, and disk usage.

```bash
python scripts/nn_input/inspect_nn_input_cache.py
```

## visualize_nn_input.py
Visualize NNInput outputs over 2D state slices; supports presets in [config/visualizations.yaml](../../config/visualizations.yaml) and an interactive mode.

```bash
# Static mode (saves images)
python scripts/nn_input/visualize_nn_input.py \
  --tag {NN_TAG} \
  --preset {PRESET} \
  --save-dir outputs/visualizations/nn_inputs/{NN_TAG}/{PRESET}

# Interactive mode (slider UI)
python scripts/nn_input/visualize_nn_input.py \
  --tag {NN_TAG} \
  --preset {PRESET} \
  --interactive
```

Outputs are written under `outputs/visualizations/nn_inputs/{TAG}/` by default.
