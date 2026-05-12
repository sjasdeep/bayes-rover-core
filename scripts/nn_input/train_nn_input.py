#!/usr/bin/env python3
"""
Train a PyTorch NN to approximate an Input(state[, time]) mapping.

Sources:
    - Input class implementation (e.g., ZeroInput)
    - Cached GridInput (via tag)

Config:
    - source.input_class: ZeroInput | GridInput
    - source.input_tag: TAG (required if input_class == GridInput)
    - source.type: control | disturbance | uncertainty
    - num_samples: int (if omitted and GridInput -> use ALL grid points; if omitted and not GridInput -> error)

Outputs:
    - .cache/nn_inputs/{TAG}.pth and .cache/nn_inputs/{TAG}.meta.json
    - .meta.json contains sizes, ranges, periodic dims, time_invariant, system_name, input_type, input_class/input_tag
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import os
import sys
import math
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.utils.registry import instantiate_system
from src.utils.config import load_nn_training_config
from src.utils.cache_loaders import resolve_input_with_class_and_tag
from src.impl.inputs.derived.grid_input import GridInput
from src.core.inputs import Input
from src.core.systems import System
from src.utils.nn import MLP


def _build_from_grid_input(
    gi: GridInput,
    system: System,
    *,
    num_samples: Optional[int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Access grid points
    state_axes: List[torch.Tensor] = gi._state_grid_points  # type: ignore[attr-defined]
    time_axis: Optional[torch.Tensor] = gi._time_grid_points  # type: ignore[attr-defined]

    # Build state mesh and flatten
    mesh = torch.meshgrid(*state_axes, indexing='ij')
    state_pts = torch.stack([m.reshape(-1) for m in mesh], dim=-1).to(device)

    # Build full dataset across state-time grid
    if (not gi.time_invariant) and (time_axis is not None):
        t = time_axis.to(device)
        S = state_pts.shape[0]
        T = t.shape[0]
        state_rep = state_pts.repeat_interleave(T, dim=0)
        time_rep = t.repeat(S)
        X_full = torch.cat([state_rep, time_rep.unsqueeze(-1)], dim=-1)
        with torch.no_grad():
            Ys = []
            for ti in t.tolist():
                Ys.append(gi.input(state_pts, float(ti)))
            Y_full = torch.cat(Ys, dim=0)
    else:
        X_full = state_pts
        with torch.no_grad():
            Y_full = gi.input(state_pts, 0.0)

    # If num_samples specified, subsample uniformly from full dataset
    if num_samples is not None and X_full.shape[0] > num_samples:
        idx = torch.randperm(X_full.shape[0])[:num_samples]
        X_full = X_full[idx]
        Y_full = Y_full[idx]

    return X_full.to(device), Y_full.to(device)


def _build_from_input(
    inp: Input,
    system: System,
    *,
    samples: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Uniform state sampling within system limits
    low = system.state_limits[0].to(device)
    high = system.state_limits[1].to(device)
    u = torch.rand(samples, system.state_dim, device=device)
    states = low + u * (high - low)

    if not getattr(inp, 'time_invariant', True):
        # Single source of truth: always use System.time_horizon
        t_min = 0.0
        t_max = float(getattr(system, 'time_horizon'))
        # Uniform time samples in [t_min, t_max]
        t = t_min + torch.rand(samples, 1, device=device) * max(t_max - t_min, 1e-6)
        X = torch.cat([states, t], dim=-1)
        with torch.no_grad():
            Ys = []
            for i in range(samples):
                Ys.append(inp.input(states[i:i+1], float(t[i].item())))
            Y = torch.cat(Ys, dim=0)
    else:
        X = states
        with torch.no_grad():
            Y = inp.input(states, 0.0)

    return X.to(device), Y.to(device)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Train a NN to approximate an Input(state[, time]).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Config-driven usage:
  Training parameters come from config/nn_training.yaml.
  Use --set KEY=VALUE for ad-hoc overrides.

Examples:
  python train_nn_input.py --system RoverDark --input-class RoverDark_MPC --input-tag RoverDark_MPC --tag RoverDark_MPC_NN
  python train_nn_input.py --system RoverDark --preset default --tag my_nn --set epochs=200 --set hidden=256
""",
    )
    # Essential args
    p.add_argument('--system', required=True, help='System class name')
    p.add_argument('--tag', required=False, help='Cache tag to save under .cache/nn_inputs/{TAG}')
    p.add_argument('--config', default='config/nn_training.yaml', help='YAML config path')
    p.add_argument('--preset', default=None, help='Optional preset name within config')
    
    # Generic override mechanism
    p.add_argument('--set', type=str, action='append', dest='overrides', metavar='KEY=VALUE',
                   help='Override config value (can be repeated). Examples: --set epochs=200 --set hidden=128')
    
    # Convenience args (shortcuts for --set)
    p.add_argument('--input-class', dest='input_class', help="Input class (convenience for --set source.input_class=X)")
    p.add_argument('--input-tag', dest='input_tag', help="Cache tag (convenience for --set source.input_tag=X)")
    p.add_argument('--type', choices=['control', 'disturbance', 'uncertainty'], default=None, help='Channel type (convenience for --set source.type=X)')
    p.add_argument('--num-samples', type=int, default=None, help='Sample count (convenience for --set num_samples=X)')
    p.add_argument('--hidden', type=int, default=None, help='Hidden layer width (convenience for --set hidden=X)')
    p.add_argument('--layers', type=int, default=None, help='Number of hidden layers (convenience for --set layers=X)')
    p.add_argument('--epochs', type=int, default=None, help='Training epochs (convenience for --set epochs=X)')
    p.add_argument('--batch-size', type=int, default=None, help='Batch size (convenience for --set batch_size=X)')
    p.add_argument('--lr', type=float, default=None, help='Learning rate (convenience for --set lr=X)')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--description', required=False, help='Description (convenience for --set description=X)')
    
    # Logging (optional)
    p.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    p.add_argument('--wandb-project', default=None, help='W&B project name')
    p.add_argument('--wandb-entity', default=None, help='W&B entity')
    p.add_argument('--wandb-mode', default=None, choices=['online', 'offline'], help='W&B mode')
    p.add_argument('--wandb-tags', nargs='*', default=None, help='W&B tags')
    
    # Checkpointing
    p.add_argument('--checkpoint-dir', default=None, help='Checkpoint directory')
    p.add_argument('--resume-from', dest='resume_from', default=None, help='Resume from checkpoint')
    p.add_argument('--no-checkpoints', action='store_true', help='Disable checkpointing')
    args = p.parse_args()

    device = torch.device(args.device)
    system = instantiate_system(args.system)

    # Load config and merge
    cfg = load_nn_training_config(args.system, preset=args.preset, path=args.config)
    
    # Apply CLI convenience args as overrides
    from src.utils.config import parse_key_value_overrides, apply_overrides
    cli_overrides = {}
    if args.input_class:
        cli_overrides.setdefault('source', {})['input_class'] = args.input_class
    if args.input_tag:
        cli_overrides.setdefault('source', {})['input_tag'] = args.input_tag
    if args.type:
        cli_overrides.setdefault('source', {})['type'] = args.type
    if args.num_samples is not None:
        cli_overrides['num_samples'] = args.num_samples
    if args.hidden is not None:
        cli_overrides['hidden'] = args.hidden
    if args.layers is not None:
        cli_overrides['layers'] = args.layers
    if args.epochs is not None:
        cli_overrides['epochs'] = args.epochs
    if args.batch_size is not None:
        cli_overrides['batch_size'] = args.batch_size
    if args.lr is not None:
        cli_overrides['lr'] = args.lr
    if args.description:
        cli_overrides['description'] = args.description
    if args.tag:
        cli_overrides['tag'] = args.tag
    
    if cli_overrides:
        # Deep merge for nested source dict
        for k, v in cli_overrides.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k] = {**cfg.get(k, {}), **v}
            else:
                cfg[k] = v
    
    # Apply --set overrides (highest priority)
    if args.overrides:
        set_overrides = parse_key_value_overrides(args.overrides)
        cfg = apply_overrides(cfg, set_overrides)
        print(f"Applied --set overrides: {set_overrides}")

    # Resolve source settings from merged config
    src = cfg.get('source', {}) if cfg else {}
    source_class = src.get('input_class')
    resolved_grid_tag = src.get('input_tag')
    resolved_type = src.get('type')
    if not resolved_type:
        raise ValueError("source.type must be specified via --set source.type=X or config source.type")
    if not source_class:
        raise ValueError('Input source requires --set source.input_class=X or source.input_class in config')

    # Build source Input using consolidated resolver
    source = resolve_input_with_class_and_tag(
        system,
        input_class=source_class,
        tag=resolved_grid_tag,
        role=resolved_type,
        device=str(device) if device else None,
    )

    # Determine unified sampling from merged config
    num_samples = cfg.get('num_samples')

    # Assemble dataset using unified semantics
    if isinstance(source, GridInput):
        X, Y = _build_from_grid_input(
            source, system,
            num_samples=num_samples,
            device=device,
        )
    else:
        if num_samples is None:
            raise ValueError("num_samples is required when training from a non-GridInput source. Use --set num_samples=N")
        X, Y = _build_from_input(
            source, system,
            samples=num_samples,
            device=device,
        )
    # Compute ranges for normalization
    x_min = X.min(dim=0).values
    x_max = X.max(dim=0).values
    y_min = Y.min(dim=0).values
    y_max = Y.max(dim=0).values

    # Periodic dims from system; time (if present) is treated as non-periodic appended last
    periodic_dims = [i for i, is_per in enumerate(system.state_periodic) if is_per]

    input_dim = X.shape[-1]
    output_dim = Y.shape[-1]
    
    # Get model/training params from merged config (supports flat or nested keys)
    def _get_cfg(key, nested_path=None, default=None):
        """Get config value, checking both flat key and nested path."""
        if key in cfg:
            return cfg[key]
        if nested_path:
            parts = nested_path.split('.')
            val = cfg
            for p in parts:
                if isinstance(val, dict) and p in val:
                    val = val[p]
                else:
                    return default
            return val
        return default
    
    hidden = _get_cfg('hidden', 'model.hidden')
    if hidden is None:
        raise ValueError("hidden must be provided via --set hidden=X or config model.hidden")
    hidden = int(hidden)
    
    layers = _get_cfg('layers', 'model.layers')
    if layers is None:
        raise ValueError("layers must be provided via --set layers=X or config model.layers")
    layers = int(layers)
    
    sizes = [input_dim, *([hidden] * layers), output_dim]
    model = MLP(
        sizes,
        input_min=x_min.tolist(),
        input_max=x_max.tolist(),
        output_min=y_min.tolist(),
        output_max=y_max.tolist(),
        periodic_input_dims=periodic_dims,
        device=device,
    ).to(device)

    # Training params from merged config
    batch_size = _get_cfg('batch_size', 'train.batch_size')
    if batch_size is None:
        raise ValueError("batch_size must be provided via --set batch_size=X or config train.batch_size")
    batch_size = int(batch_size)
    
    lr = _get_cfg('lr', 'train.lr')
    if lr is None:
        raise ValueError("lr must be provided via --set lr=X or config train.lr")
    lr = float(lr)
    
    ds = TensorDataset(X, Y)
    
    epochs = _get_cfg('epochs', 'train.epochs')
    if epochs is None:
        raise ValueError("epochs must be provided via --set epochs=X or config train.epochs")
    epochs = int(epochs)
    
    # Choose dataloading strategy:
    # - If tensors are already on CUDA, bypass DataLoader to avoid host<->device overhead
    # - If tensors are on CPU, use DataLoader with sensible defaults
    on_cuda = (X.is_cuda and Y.is_cuda)
    dl: Optional[DataLoader] = None
    if on_cuda:
        dl = None  # manual fast path below
    else:
        # CPU-resident dataset: enable pinned memory and modest workers for better throughput
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=os.cpu_count() if os.cpu_count() is not None else 0,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
        )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Resolve tag early for logging and checkpoint directory defaults
    save_tag = args.tag or (cfg.get('tag') if isinstance(cfg, dict) else None)

    # Optional: initialize Weights & Biases logging (after we know sizes and hyperparams)
    enable_wandb = bool(args.wandb or (cfg.get('logging', {}).get('wandb', {}).get('enable') if isinstance(cfg, dict) and cfg.get('logging') else False))
    wandb_run = None
    if enable_wandb:
        import wandb  # type: ignore
        wb_cfg = cfg.get('logging', {}).get('wandb', {}) if isinstance(cfg, dict) else {}
        wb_project = args.wandb_project or wb_cfg.get('project')
        if not wb_project:
            raise ValueError("wandb enabled, but no project specified (use --wandb-project or logging.wandb.project)")
        wb_entity = args.wandb_entity or wb_cfg.get('entity')
        wb_mode = args.wandb_mode or wb_cfg.get('mode')
        wb_tags = args.wandb_tags or wb_cfg.get('tags')
        wandb_config = {
            'system': args.system,
            'input_type': resolved_type,
            'input_class': source_class,
            'input_tag': (resolved_grid_tag if resolved_grid_tag else None),
            'num_samples': num_samples,
            'model.sizes': sizes,
            'train.batch_size': batch_size,
            'train.lr': lr,
            'train.epochs': epochs,
            'dataset.size': len(ds),
            'device': str(device),
            'time_invariant': bool(getattr(source, 'time_invariant', True)),
            'save_tag': save_tag,
        }
        init_kwargs = {
            'project': wb_project,
            'config': wandb_config,
            'name': f"{args.system}:{resolved_type}:{save_tag}" if save_tag else None,
        }
        if wb_tags is not None:
            init_kwargs['tags'] = wb_tags
        if wb_entity:
            init_kwargs['entity'] = wb_entity
        if wb_mode in ('online', 'offline'):
            init_kwargs['mode'] = wb_mode
        wandb_run = wandb.init(**init_kwargs)

    # ---------- Checkpoint configuration ----------
    ckpt_cfg = (cfg.get('train', {}).get('checkpoint', {}) if isinstance(cfg, dict) else {})
    ckpt_enable = (not args.no_checkpoints) and bool(ckpt_cfg.get('enable', True))
    ckpt_freq = int(ckpt_cfg.get('frequency', 1)) if ckpt_enable else 0
    ckpt_save_best = bool(ckpt_cfg.get('save_best', True)) if ckpt_enable else False
    ckpt_save_last = bool(ckpt_cfg.get('save_last', True)) if ckpt_enable else False
    ckpt_save_history = bool(ckpt_cfg.get('save_history', False)) if ckpt_enable else False
    ckpt_keep_last_n = int(ckpt_cfg.get('keep_last_n', 0)) if ckpt_enable else 0
    # Resolve checkpoint directory
    if ckpt_enable:
        default_ckpt_dir = None
        if save_tag is not None:
            default_ckpt_dir = Path('.cache') / 'nn_inputs' / 'checkpoints' / args.system / save_tag
        ckpt_dir = Path(args.checkpoint_dir or ckpt_cfg.get('dir') or (str(default_ckpt_dir) if default_ckpt_dir is not None else ''))
        if str(ckpt_dir) == '':
            print('[checkpoint] No tag or directory provided; disabling checkpoints for this run.')
            ckpt_enable = False
        else:
            ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Resume setup
    start_epoch = 1
    best_mse = math.inf
    if ckpt_enable:
        resume_path = None
        if args.resume_from:
            resume_path = Path(args.resume_from)
        elif bool(ckpt_cfg.get('resume', False)):
            explicit = ckpt_cfg.get('path')
            if explicit:
                resume_path = Path(str(explicit))
            else:
                candidate = ckpt_dir / 'last.pth'
                if candidate.exists():
                    resume_path = candidate
        if resume_path is not None and resume_path.exists():
            print(f"[checkpoint] Resuming from {resume_path}")
            ckpt_obj = torch.load(resume_path, map_location=device)
            model.load_state_dict(ckpt_obj['model_state'])
            try:
                opt.load_state_dict(ckpt_obj['optimizer_state'])
            except Exception as e:
                print(f"[checkpoint] Warning: could not load optimizer state: {e}")
            start_epoch = int(ckpt_obj.get('epoch', 0)) + 1
            best_mse = float(ckpt_obj.get('best_mse', math.inf))

    def _save_checkpoint(epoch: int, epoch_mse: float, is_best: bool = False) -> None:
        if not ckpt_enable:
            return
        state = {
            'epoch': epoch,
            'best_mse': min(best_mse, epoch_mse),
            'model_state': model.state_dict(),
            'optimizer_state': opt.state_dict(),
            'sizes': sizes,
            'metrics': {'train_mse': epoch_mse},
            'input_min': x_min.tolist(),
            'input_max': x_max.tolist(),
            'output_min': y_min.tolist(),
            'output_max': y_max.tolist(),
            'periodic_input_dims': periodic_dims,
            'time_invariant': bool(getattr(source, 'time_invariant', True)),
            'system_name': args.system,
            'input_type': resolved_type,
            'input_class': source_class,
            'input_tag': (resolved_grid_tag if resolved_grid_tag else None),
            'tag': save_tag,
        }
        if ckpt_save_last:
            torch.save(state, ckpt_dir / 'last.pth')
        if ckpt_save_history:
            ep_path = ckpt_dir / f'epoch-{epoch:04d}.pth'
            torch.save(state, ep_path)
            if ckpt_keep_last_n > 0:
                # prune older epoch files
                ep_files = sorted([p for p in ckpt_dir.glob('epoch-*.pth')])
                if len(ep_files) > ckpt_keep_last_n:
                    for pth in ep_files[0:len(ep_files) - ckpt_keep_last_n]:
                        try:
                            pth.unlink()
                        except FileNotFoundError:
                            pass
        if ckpt_save_best and is_best:
            torch.save(state, ckpt_dir / 'best.pth')

        # Mirror the latest checkpoint state into the NNInput tag cache so downstream tools can use it early
        # Only if a save tag is provided
        if save_tag:
            base = Path('.cache') / 'nn_inputs' / save_tag
            base.parent.mkdir(parents=True, exist_ok=True)
            raw_path = base.with_suffix('.pth')
            meta_path = base.with_suffix('.meta.json')
            # Save the model state_dict in NNInput compatible format
            torch.save({'state_dict': model.state_dict(), 'sizes': sizes}, raw_path)
            # Write metadata including checkpoint info
            from datetime import datetime, timezone
            meta = {
                'sizes': sizes,
                'input_min': x_min.tolist(),
                'input_max': x_max.tolist(),
                'output_min': y_min.tolist(),
                'output_max': y_max.tolist(),
                'periodic_input_dims': periodic_dims,
                'time_invariant': bool(getattr(source, 'time_invariant', True)),
                'system_name': args.system,
                'input_type': resolved_type,
                'input_class': source_class,
                'input_tag': (resolved_grid_tag if resolved_grid_tag else None),
                'description': (args.description if args.description is not None else cfg.get('description', '')),
                'created_at': datetime.now(timezone.utc).isoformat(),
                # New: checkpoint awareness for early consumption
                'training_in_progress': True,
                'checkpoint': {
                    'epoch': int(epoch),
                    'best_mse': float(min(best_mse, epoch_mse)),
                },
            }
            try:
                with open(meta_path, 'w') as f:
                    json.dump(meta, f)
            except Exception as e:
                print(f"[checkpoint->nn_cache] Warning: could not write NN cache metadata: {e}")

    model.train()
    # Simple throughput meter (samples/sec per epoch)
    import time
    pbar = tqdm(range(start_epoch, epochs + 1), desc="Training", unit="epoch")
    for epoch in pbar:
        t0 = time.perf_counter()
        total = 0.0
        count = 0
        if dl is not None:
            # Standard DataLoader path (CPU dataset)
            for xb, yb in dl:
                # Move to device if needed (non_blocking for pinned memory)
                if xb.device != device:
                    xb = xb.to(device, non_blocking=True)
                if yb.device != device:
                    yb = yb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
                total += float(loss.item()) * xb.shape[0]
                count += xb.shape[0]
        else:
            # Fast on-device batching path (CUDA dataset)
            N = X.shape[0]
            # On-device random permutation for shuffle
            perm = torch.randperm(N, device=device)
            # Iterate in contiguous chunks of batch_size
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                idx = perm[start:end]
                xb = X.index_select(0, idx)
                yb = Y.index_select(0, idx)
                opt.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
                total += float(loss.item()) * xb.shape[0]
                count += xb.shape[0]
        dt = max(time.perf_counter() - t0, 1e-9)
        sps = count / dt
        epoch_mse = total / max(count, 1)
        pbar.set_postfix(MSE=f"{epoch_mse:.6f}", sps=f"{sps:.0f}")
        if wandb_run is not None:
            wandb.log({'train/mse': epoch_mse, 'epoch': epoch}, step=epoch)
        # Checkpoint save
        if ckpt_enable and (ckpt_freq > 0) and ((epoch % ckpt_freq == 0) or (epoch == epochs)):
            improved = epoch_mse < best_mse - 1e-12
            if improved:
                best_mse = epoch_mse
            _save_checkpoint(epoch, epoch_mse, is_best=improved)

        # ---------- Visualization to W&B ----------
        if enable_wandb and wandb_run is not None:
            wb_cfg = cfg.get('logging', {}).get('wandb', {}) if isinstance(cfg, dict) else {}
            vis_cfg = wb_cfg.get('visualize', {}) if isinstance(wb_cfg, dict) else {}
            vis_enable = bool(vis_cfg.get('enable', False))
            if vis_enable:
                every = int(vis_cfg.get('every_n_epochs', 10))
                # Only run at the defined cadence and at the final epoch
                if (epoch % max(every, 1) == 0) or (epoch == epochs):
                    state_axes = vis_cfg.get('state_axes', [0, 1])
                    if not isinstance(state_axes, (list, tuple)) or len(state_axes) != 2:
                        state_axes = [0, 1]
                    ax_i, ax_j = int(state_axes[0]), int(state_axes[1])
                    if system.state_dim < 2:
                        # Not enough dims to plot a 2D slice
                        pass
                    else:
                        grid_res = vis_cfg.get('grid_resolution', 101)
                        if isinstance(grid_res, (list, tuple)) and len(grid_res) == 2:
                            nx, ny = int(grid_res[0]), int(grid_res[1])
                        else:
                            nx = ny = int(grid_res)
                        # Choose time slice
                        expects_time = (model.sizes[0] == system.state_dim + 1) if hasattr(model, 'sizes') else (X.shape[-1] == system.state_dim + 1)
                        t_val = 0.0
                        if not getattr(source, 'time_invariant', True) and expects_time:
                            t_cfg = vis_cfg.get('time', 'mid')
                            if isinstance(t_cfg, str) and t_cfg.lower() == 'mid':
                                t_val = float(getattr(system, 'time_horizon')) / 2.0
                            else:
                                try:
                                    t_val = float(t_cfg)
                                except Exception:
                                    t_val = float(getattr(system, 'time_horizon')) / 2.0

                        # Build a 2D grid slice
                        if isinstance(source, GridInput):
                            # Use exact grid points to avoid off-grid lookup errors
                            try:
                                gi_state_axes: List[torch.Tensor] = source._state_grid_points  # type: ignore[attr-defined]
                            except Exception:
                                gi_state_axes = []
                            if gi_state_axes and ax_i < len(gi_state_axes) and ax_j < len(gi_state_axes):
                                xi = gi_state_axes[ax_i].to(device)
                                yj = gi_state_axes[ax_j].to(device)
                                nx, ny = int(xi.numel()), int(yj.numel())
                                mesh_i, mesh_j = torch.meshgrid(xi, yj, indexing='ij')
                                # For non-plotted dims, pick middle grid entry to stay on-grid
                                base = []
                                for d, ax in enumerate(gi_state_axes):
                                    mid_idx = int(ax.numel() // 2)
                                    base.append(ax[mid_idx].to(device))
                                base = torch.stack(base) if len(base) > 0 else torch.empty(0, device=device)
                                pts = base.repeat(nx * ny, 1)
                                pts[:, ax_i] = mesh_i.reshape(-1)
                                pts[:, ax_j] = mesh_j.reshape(-1)
                            else:
                                # Fallback to continuous slice if grid axes not available
                                low = system.state_limits[0].to(device)
                                high = system.state_limits[1].to(device)
                                mids = (low + high) / 2.0
                                xi = torch.linspace(low[ax_i].item(), high[ax_i].item(), steps=nx, device=device)
                                yj = torch.linspace(low[ax_j].item(), high[ax_j].item(), steps=ny, device=device)
                                mesh_i, mesh_j = torch.meshgrid(xi, yj, indexing='ij')
                                pts = mids.repeat(nx * ny, 1)
                                pts[:, ax_i] = mesh_i.reshape(-1)
                                pts[:, ax_j] = mesh_j.reshape(-1)
                        else:
                            low = system.state_limits[0].to(device)
                            high = system.state_limits[1].to(device)
                            mids = (low + high) / 2.0
                            xi = torch.linspace(low[ax_i].item(), high[ax_i].item(), steps=nx, device=device)
                            yj = torch.linspace(low[ax_j].item(), high[ax_j].item(), steps=ny, device=device)
                            mesh_i, mesh_j = torch.meshgrid(xi, yj, indexing='ij')
                            pts = mids.repeat(nx * ny, 1)
                            pts[:, ax_i] = mesh_i.reshape(-1)
                            pts[:, ax_j] = mesh_j.reshape(-1)

                        # Prepare model inputs (append time if necessary)
                        model_in = pts
                        if expects_time:
                            t_col = torch.full((pts.shape[0], 1), float(t_val), device=device)
                            model_in = torch.cat([pts, t_col], dim=-1)

                        with torch.no_grad():
                            pred = model(model_in).detach()
                            # Ground-truth at t_val; if GridInput, we constructed on-grid points above
                            if getattr(source, 'time_invariant', True):
                                truth = source.input(pts, 0.0)
                            else:
                                # If GridInput has discrete time grid, choose nearest grid point when using grid axes
                                if isinstance(source, GridInput):
                                    try:
                                        t_axis: Optional[torch.Tensor] = source._time_grid_points  # type: ignore[attr-defined]
                                    except Exception:
                                        t_axis = None
                                    if t_axis is not None and t_axis.numel() > 0:
                                        # pick middle by default; if user provided numeric time, snap to nearest
                                        if isinstance(vis_cfg.get('time', 'mid'), (int, float)):
                                            t_scalar = float(vis_cfg.get('time'))
                                            # snap to nearest grid time
                                            idx = int((t_axis - t_scalar).abs().argmin().item())
                                        else:
                                            idx = int(t_axis.numel() // 2)
                                        t_snap = float(t_axis[idx].item())
                                        truth = source.input(pts, t_snap)
                                    else:
                                        truth = source.input(pts, float(t_val))
                                else:
                                    truth = source.input(pts, float(t_val))

                        # Select output index
                        out_idx = int(vis_cfg.get('output_index', 0))
                        out_idx = max(0, min(out_idx, pred.shape[-1] - 1))
                        z_pred = pred[:, out_idx].reshape(nx, ny).detach().cpu().numpy()
                        z_true = truth[:, out_idx].reshape(nx, ny).detach().cpu().numpy()
                        z_err = z_pred - z_true

                        # Matplotlib figure using same style as visualize_grid_input.py (contourf, RdBu)
                        try:
                            # Ensure a non-interactive backend for headless environments (e.g., SSH/CI)
                            if 'matplotlib.pyplot' not in sys.modules:
                                import matplotlib  # type: ignore
                                backend_opt = vis_cfg.get('matplotlib_backend', None)
                                if backend_opt is not None:
                                    try:
                                        matplotlib.use(str(backend_opt))
                                    except Exception:
                                        matplotlib.use('Agg')
                                else:
                                    matplotlib.use('Agg')
                            import matplotlib.pyplot as plt  # type: ignore

                            # Determine color scale from control limits across the slice (match visualize_grid_input)
                            # Use snapped time if available
                            t_for_limits = float(t_val)
                            if not getattr(source, 'time_invariant', True) and isinstance(source, GridInput):
                                try:
                                    t_axis: Optional[torch.Tensor] = source._time_grid_points  # type: ignore[attr-defined]
                                except Exception:
                                    t_axis = None
                                if t_axis is not None and t_axis.numel() > 0:
                                    # If numeric value provided, snap; else mid
                                    if isinstance(vis_cfg.get('time', 'mid'), (int, float)):
                                        t_scalar = float(vis_cfg.get('time'))
                                        idx = int((t_axis - t_scalar).abs().argmin().item())
                                    else:
                                        idx = int(t_axis.numel() // 2)
                                    t_for_limits = float(t_axis[idx].item())

                            flat_states_cpu = pts.reshape(-1, pts.shape[-1]).detach().cpu()
                            try:
                                ctrl_lower, ctrl_upper = system.control_limits(flat_states_cpu, t_for_limits)
                                vmin = float(ctrl_lower[:, out_idx].min().item())
                                vmax = float(ctrl_upper[:, out_idx].max().item())
                            except Exception:
                                # Fallback to data-driven scale if system limits not available
                                vmin = float(min(z_true.min(), z_pred.min()))
                                vmax = float(max(z_true.max(), z_pred.max()))

                            # Build coordinate grids for contourf
                            V1 = mesh_i.detach().cpu().numpy()
                            V2 = mesh_j.detach().cpu().numpy()

                            fig, axs = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
                            # Truth
                            c0 = axs[0].contourf(V1, V2, z_true, levels=20, cmap='RdBu', vmin=vmin, vmax=vmax)
                            # Prediction (share vmin/vmax with truth)
                            c1 = axs[1].contourf(V1, V2, z_pred, levels=20, cmap='RdBu', vmin=vmin, vmax=vmax)
                            # Error: symmetric bounds derived from control-limit magnitude for consistency
                            emax = max(abs(vmin), abs(vmax))
                            c2 = axs[2].contourf(V1, V2, z_err, levels=20, cmap='coolwarm', vmin=-emax, vmax=emax)

                            # Labels using state labels if available
                            lbl_i = system.state_labels[ax_i] if hasattr(system, 'state_labels') and ax_i < len(system.state_labels) else f'state[{ax_i}]'
                            lbl_j = system.state_labels[ax_j] if hasattr(system, 'state_labels') and ax_j < len(system.state_labels) else f'state[{ax_j}]'
                            titles = ['Truth', 'Prediction', 'Error (pred - truth)']
                            for ax_plot, title in zip(axs, titles):
                                ax_plot.set_xlabel(lbl_i)
                                ax_plot.set_ylabel(lbl_j)
                                ax_plot.set_title(title)
                                ax_plot.grid(True, alpha=0.3)

                            # Colorbars
                            fig.colorbar(c0, ax=axs[0], fraction=0.046, pad=0.04)
                            fig.colorbar(c1, ax=axs[1], fraction=0.046, pad=0.04)
                            fig.colorbar(c2, ax=axs[2], fraction=0.046, pad=0.04)

                            wandb.log({
                                'viz/state_slice': wandb.Image(fig),
                                'viz/state_axes': state_axes,
                                'viz/output_index': out_idx,
                                'viz/time': float(t_for_limits),
                            }, step=epoch)
                            plt.close(fig)
                        except Exception as e:
                            print(f"[wandb/visualize] Skipped visualization due to: {e}")

    # Save raw state_dict and metadata (usable by NNInput and tools like auto_LiRPA)
    model.eval()
    # Resolve output destination (tag is required either via CLI or config)
    tag = save_tag
    if not tag:
        raise ValueError('Tag is required. Provide --tag or set tag in config to save into .cache/nn_inputs/{TAG}.')
    out_path = Path('.cache') / 'nn_inputs' / tag
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = out_path.with_suffix('.pth')
    meta_path = out_path.with_suffix('.meta.json')
    torch.save({'state_dict': model.state_dict(), 'sizes': sizes}, raw_path)
    # Build rich metadata
    from datetime import datetime, timezone
    meta = {
        'sizes': sizes,
        'input_min': x_min.tolist(),
        'input_max': x_max.tolist(),
        'output_min': y_min.tolist(),
        'output_max': y_max.tolist(),
        'periodic_input_dims': periodic_dims,
        'time_invariant': bool(getattr(source, 'time_invariant', True)),
        'system_name': args.system,
        'input_type': resolved_type,
        'input_class': source_class,
        'input_tag': (resolved_grid_tag if resolved_grid_tag else None),
        'description': (args.description if args.description is not None else cfg.get('description', '')),
        'created_at': datetime.now(timezone.utc).isoformat(),
        # Include final checkpoint info for traceability
        'training_in_progress': False,
        'checkpoint': {
            'epoch': int(epochs),
            'best_mse': float(best_mse if math.isfinite(best_mse) else -1.0),
        },
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f)
    print(f"Saved NNInput cache tag '{tag}' to {raw_path} and {meta_path}")

    # Close W&B run if active
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == '__main__':
    main()
