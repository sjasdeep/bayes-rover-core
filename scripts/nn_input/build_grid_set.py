#!/usr/bin/env python3
"""
Build a tagged GridSet cache for a specific system and NNInput cache tag.

This constructs a state (and optional time) grid from config/resolutions.yaml,
and at each grid point computes a box set over the NN output using interval
bound propagation via auto_LiRPA. The input uncertainty is modeled as a box in
state space given by the system's uncertainty_limits around the grid center.

Only the Box set case is implemented. Corner provenance or state estimates are
not saved (auto_LiRPA does not directly provide extremal inputs achieving the
bounds).

Usage:
  python scripts/nn_input/build_grid_set.py \
      --system RoverDark \
      --nn-input-tag RoverDark_MPC_NN \
      --tag RoverDark_MPC_NN_Set \
      --description "NNInput box set via LiRPA" \
      --batch-size 4096 --method CROWN-IBP
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from itertools import product

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Try to import auto_LiRPA; fall back to local libraries path if needed
try:
    from auto_LiRPA import BoundedModule, BoundedTensor  # type: ignore
    from auto_LiRPA.perturbations import PerturbationLpNorm  # type: ignore
except Exception:
    LIB_PATH = PROJECT_ROOT / 'libraries' / 'auto_LiRPA'
    if str(LIB_PATH) not in sys.path:
        sys.path.insert(0, str(LIB_PATH))
    from auto_LiRPA import BoundedModule, BoundedTensor  # type: ignore
    from auto_LiRPA.perturbations import PerturbationLpNorm  # type: ignore

from src.utils.cache_loaders import (
    get_nn_input_metadata,
    instantiate_system_by_name,
    load_nn_input_by_tag,
)
from src.utils.config import load_resolution_config


def _build_state_grid_points(system, state_resolution: List[int]) -> List[torch.Tensor]:
    if len(state_resolution) != system.state_dim:
        raise ValueError(
            f"State resolution length {len(state_resolution)} does not match system state dimension {system.state_dim}"
        )
    pts: List[torch.Tensor] = []
    for i, res in enumerate(state_resolution):
        lo = float(system.state_limits[0, i].item())
        hi = float(system.state_limits[1, i].item())
        if torch.isinf(torch.tensor(lo)) or torch.isinf(torch.tensor(hi)):
            raise ValueError(f"State dimension {i} has infinite limits; cannot build grid")
        pts.append(torch.linspace(lo, hi, int(res)))
    return pts


@torch.no_grad()
def _compute_bounds_for_batch(
    li_model: BoundedModule,
    x_center: torch.Tensor,
    x_L: torch.Tensor,
    x_U: torch.Tensor,
    *,
    method: str = 'CROWN-IBP',
) -> Tuple[torch.Tensor, torch.Tensor]:
    pert = PerturbationLpNorm(norm=np.inf, x_L=x_L, x_U=x_U)
    x_b = BoundedTensor(x_center, pert)
    lb, ub = li_model.compute_bounds(x=(x_b,), method=method)
    # Numerical safety: ensure lower <= upper elementwise
    lb_s = torch.minimum(lb, ub)
    ub_s = torch.maximum(lb, ub)
    return lb_s, ub_s


def build_grid_set_from_nn(
    system_name: str,
    nn_input_tag: str,
    tag: str,
    description: str = '',
    *,
    config_path: str = 'config/resolutions.yaml',
    time_resolution: Optional[int] = None,
    batch_size: int = 4096,
    method: str = 'CROWN-IBP',
    device: Optional[str] = None,
    force: bool = False,
    # Uncertainty splitting options to tighten bounds
    split_dims: Optional[List[int]] = None,
    splits: Optional[List[int]] = None,
) -> bool:
    # Instantiate system and NNInput
    try:
        system = instantiate_system_by_name(system_name)
    except Exception as e:
        print(f"Error: System '{system_name}' not found or failed to instantiate: {e}")
        return False

    meta = get_nn_input_metadata(nn_input_tag)
    if meta.get('system_name') != system_name:
        print(f"Error: NNInput tag '{nn_input_tag}' was built for system '{meta.get('system_name')}', not '{system_name}'")
        if meta.get('system_name') == 'RoverDark' and 'RoverDark' in system_name:
            print(f'Overriding system mismatch...')
        else:
            return False
    nn_input = load_nn_input_by_tag(nn_input_tag, system, device=torch.device('cuda' if (device in (None, 'cuda') and torch.cuda.is_available()) else 'cpu'))

    # Resolve grid config (use 'default' node for NN input)
    try:
        res_cfg = load_resolution_config(system_name, None, config_path)
    except Exception as e:
        print(f"Error: Failed to load resolutions from {config_path}: {e}")
        return False
    if 'state_resolution' not in res_cfg:
        print("Error: resolutions.yaml missing 'state_resolution' for this system")
        return False
    state_grid_points = _build_state_grid_points(system, res_cfg['state_resolution'])

    # Time grid: include time dimension if either the NN is time-varying OR
    # the system's uncertainty limits are time-varying.
    nn_time_varying = not getattr(nn_input, 'time_invariant', True)
    unc_time_varying = not getattr(system, 'time_invariant_uncertainty_limits', False)
    if nn_time_varying or unc_time_varying:
        th = float(getattr(system, 'time_horizon'))
        tr = int(time_resolution if time_resolution is not None else res_cfg.get('time_resolution', 0) or 0)
        if tr <= 1:
            tr = 11  # small default if missing
        time_grid_points = torch.linspace(0.0, th, tr)
        reason = []
        if nn_time_varying:
            reason.append('NNInput')
        if unc_time_varying:
            reason.append('uncertainty_limits')
        reason_str = "/".join(reason)
        print(f"Time-varying detected ({reason_str}): synthesizing time grid with {tr} steps over horizon {th}")
    else:
        time_grid_points = None
        print("Both NNInput and uncertainty limits are time-invariant: no time grid")

    # Destination
    out_dir = Path('.cache') / 'grid_sets'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{tag}.pkl"
    if out_path.exists() and not force:
        print(f"✓ GridSet cache already exists: {out_path}")
        return True
    if out_path.exists() and force:
        print(f"⚠ Overwriting existing cache: {out_path.name}")
        out_path.unlink(missing_ok=True)

    # Shapes and allocations
    axes = [ax.detach().cpu() for ax in state_grid_points]
    shape = tuple(len(ax) for ax in axes)
    state_dim = len(axes)
    t_size = 1 if time_grid_points is None else int(time_grid_points.numel())
    out_dim = int(getattr(nn_input, 'dim', 1))
    dtype = torch.float32
    box_lower = torch.empty(*shape, t_size, out_dim, dtype=dtype)
    box_upper = torch.empty_like(box_lower)

    # Prepare LiRPA model once on target device
    in_dim = state_dim + (0 if getattr(nn_input, 'time_invariant', True) else 1)
    target_device = torch.device('cuda' if (device in (None, 'cuda') and torch.cuda.is_available()) else (device or 'cpu'))
    dummy = torch.zeros(1, in_dim, device=target_device, dtype=dtype)
    # Ensure model is on the chosen device
    nn_input._module.to(target_device)
    # Bound optimization options (can tighten bounds when enabled)
    bound_opts = getattr(build_grid_set_from_nn, '_bound_opts', None)
    li_model = BoundedModule(nn_input._module, dummy, device=target_device, bound_opts=bound_opts)
    print(f"LiRPA device: model={next(nn_input._module.parameters()).device}, compute={target_device}")

    # Iterate times
    meshes = torch.meshgrid(*axes, indexing='ij')
    grid_states = torch.stack(meshes, dim=-1).reshape(-1, state_dim)  # [N, state_dim]
    N = int(grid_states.shape[0])

    # Parse uncertainty splitting configuration
    if split_dims is not None and len(split_dims) == 0:
        split_dims = None
    split_cfg = None
    if split_dims is not None:
        if splits is None or (len(splits) not in (1, len(split_dims))):
            raise ValueError("When using --split-dims, provide --splits as a single int or one per split dim")
        if len(splits) == 1:
            splits = [int(splits[0]) for _ in split_dims]
        split_cfg = list(zip([int(d) for d in split_dims], [int(s) for s in splits]))
        if any(d < 0 or d >= state_dim for d, _ in split_cfg):
            raise ValueError(f"split dim out of range; valid 0..{state_dim-1}")
        if any(s < 2 for _, s in split_cfg):
            raise ValueError("splits must be >= 2 when provided")

    pbar = tqdm(range(t_size), desc="Computing bounds", unit="t")
    for ti in pbar:
        t_val = 0.0 if time_grid_points is None else float(time_grid_points[ti].item())
        pbar.set_description(f"t={t_val:.2f} ({N:,} states)")
        # Compute uncertainty bounds around grid centers
        e_min, e_max = system.uncertainty_limits(grid_states, float(t_val))  # [N, state_dim]
        # Build x_center, x_L, x_U
        x_center = grid_states.clone()
        x_L = x_center + e_min
        x_U = x_center + e_max
        if in_dim == state_dim + 1:
            t_col = torch.full((N, 1), float(t_val))
            x_center = torch.cat([x_center, t_col], dim=-1)
            x_L = torch.cat([x_L, t_col], dim=-1)
            x_U = torch.cat([x_U, t_col], dim=-1)

        # Move to device (and enforce dtype)
        x_center = x_center.to(device=target_device, dtype=dtype, non_blocking=True)
        x_L = x_L.to(device=target_device, dtype=dtype, non_blocking=True)
        x_U = x_U.to(device=target_device, dtype=dtype, non_blocking=True)

        # Batched bound computation (optionally with uncertainty splitting per batch)
        lbs = []
        ubs = []
        for i in range(0, N, int(batch_size)):
            xc = x_center[i:i+batch_size]
            xl = x_L[i:i+batch_size]
            xu = x_U[i:i+batch_size]
            if split_cfg is None:
                lb, ub = _compute_bounds_for_batch(li_model, xc, xl, xu, method=method)
            else:
                lb_min = None
                ub_max = None
                # Iterate all segment combinations across selected dims
                for combo in product(*[range(s) for _, s in split_cfg]):
                    xl_c = xl.clone()
                    xu_c = xu.clone()
                    for (dim, s), seg_idx in zip(split_cfg, combo):
                        delta = xu[:, dim] - xl[:, dim]
                        lower = xl[:, dim] + (float(seg_idx) / float(s)) * delta
                        upper = xl[:, dim] + (float(seg_idx + 1) / float(s)) * delta
                        xl_c[:, dim] = lower
                        xu_c[:, dim] = upper
                    lb_c, ub_c = _compute_bounds_for_batch(li_model, xc, xl_c, xu_c, method=method)
                    lb_min = lb_c if lb_min is None else torch.minimum(lb_min, lb_c)
                    ub_max = ub_c if ub_max is None else torch.maximum(ub_max, ub_c)
                lb, ub = lb_min, ub_max
            lbs.append(lb.detach().to('cpu'))
            ubs.append(ub.detach().to('cpu'))
        lb_all = torch.cat(lbs, dim=0)
        ub_all = torch.cat(ubs, dim=0)
        # Reshape to grid and assign
        lb_grid = lb_all.reshape(*shape, out_dim)
        ub_grid = ub_all.reshape(*shape, out_dim)
        box_lower[..., ti, :] = lb_grid
        box_upper[..., ti, :] = ub_grid

    # Assemble payload
    payload = {
        'tag': tag,
        'description': description,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'system_name': system_name,
        'input_name': 'NNInput',
        'set_type': 'box',
        'grid_input_tag': None,
        'nn_input_tag': nn_input_tag,
        'grid_shape': shape + (t_size,),
        'state_grid_points': [t.detach().cpu() for t in state_grid_points],
        'time_grid_points': None if time_grid_points is None else time_grid_points.detach().cpu(),
        'box_lower': box_lower,
        'box_upper': box_upper,
        # No box_state_est_corner_idx for LiRPA-based sets (extremal inputs not provided)
    }

    # Save
    import pickle
    tmp_path = out_path.with_suffix('.pkl.tmp')
    with open(tmp_path, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(out_path)
    total_points = int(np.prod(shape)) * t_size
    size_mb = (box_lower.numel() + box_upper.numel()) * box_lower.element_size() / (1024**2)
    print(f"\n✓ Saved NN-based GridSet cache: {out_path}")
    print(f"  Grid shape:   {list(shape) + [t_size, out_dim]}")
    print(f"  Grid points:  {total_points:,}")
    print(f"  Memory (MB):  {size_mb:.2f}")
    # Summarize tightness
    widths = (box_upper - box_lower).float()
    mean_w = widths.mean().item()
    max_w = widths.max().item()
    print(f"  Avg interval width: {mean_w:.6g}; Max width: {max_w:.6g}")
    return True


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Config-driven usage:
  Resolution parameters come from config/resolutions.yaml.
  Use --set KEY=VALUE for ad-hoc overrides.

Examples:
  python build_grid_set.py --system RoverDark --nn-input-tag RoverDark_MPC_NN --tag RoverDark_MPC_NN_Box
  python build_grid_set.py --system RoverDark --nn-input-tag RoverDark_MPC_NN --tag my_set --set method=IBP
""",
    )
    # Essential args
    p.add_argument('--system', type=str, required=True, help='System class name')
    p.add_argument('--nn-input-tag', type=str, required=True, help='Existing NNInput cache tag')
    p.add_argument('--tag', type=str, required=True, help='New GridSet tag')
    p.add_argument('--config', type=str, default='config/resolutions.yaml', help='Resolution config')
    p.add_argument('--force', action='store_true', help='Overwrite existing cache')
    
    # Generic override mechanism
    p.add_argument('--set', type=str, action='append', dest='overrides', metavar='KEY=VALUE',
                   help='Override config value (can be repeated)')
    
    # Convenience args (shortcuts for --set)
    p.add_argument('--description', type=str, default='', help='Description')
    p.add_argument('--time-resolution', type=int, default=None, help='Override time steps')
    p.add_argument('--batch-size', type=int, default=4096, help='Batch size for LiRPA')
    p.add_argument('--method', type=str, default='CROWN', choices=['IBP', 'CROWN', 'CROWN-IBP'], help='LiRPA bound method')
    p.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'], help='Computation device')
    
    # Advanced LiRPA options
    p.add_argument('--optimize-bound', action='store_true', default=True, help='Enable alpha-CROWN')
    p.add_argument('--optimize-iters', type=int, default=100, help='Bound optimization iterations')
    p.add_argument('--optimize-lr-alpha', type=float, default=0.5, help='Alpha learning rate')
    p.add_argument('--enable-beta-crown', action='store_true', default=True, help='Enable beta-CROWN')
    p.add_argument('--forward-refinement', action='store_true', default=True, help='Enable forward refinement')
    p.add_argument('--crown-batch-size', type=int, default=0, help='CROWN batch size (0=auto)')
    
    # Splitting options
    p.add_argument('--split-dims', type=str, default=None, help='Comma-separated state dims to split')
    p.add_argument('--splits', type=str, default=None, help='Splits per dim')
    
    # Comparison mode
    p.add_argument('--no-compare', action='store_true', help='Disable adaptive comparison')
    p.add_argument('--compare-sample', type=int, default=100000, help='Grid states for comparison')
    p.add_argument('--compare-time-resolution', type=int, default=5, help='Time steps for comparison')
    p.add_argument('--compare-batch-size', type=int, default=16384, help='Comparison batch size')
    
    args = p.parse_args()
    
    # Build config from convenience args
    from src.utils.config import parse_key_value_overrides, apply_overrides
    cfg = {
        'description': args.description,
        'time_resolution': args.time_resolution,
        'batch_size': args.batch_size,
        'method': args.method,
        'device': args.device,
    }
    
    # Apply --set overrides (highest priority)
    if args.overrides:
        set_overrides = parse_key_value_overrides(args.overrides)
        cfg = apply_overrides(cfg, set_overrides)
        print(f"Applied --set overrides: {set_overrides}")

    # Assemble bound options for BoundedModule; stash on the function so build routine can pick it up without changing signature
    bound_opts = {
        'forward_refinement': bool(args.forward_refinement),
        'crown_batch_size': (np.inf if int(args.crown_batch_size) <= 0 else int(args.crown_batch_size)),
        'optimize_bound_args': {
            'enable_alpha_crown': bool(args.optimize_bound),
            'enable_beta_crown': bool(args.enable_beta_crown),
            'iteration': int(args.optimize_iters),
            'lr_alpha': float(args.optimize_lr_alpha),
            'use_float64_in_last_iteration': True,
            'tighten_input_bounds': True,
            'best_of_oc_and_no_oc': False,
        }
    }
    # Enable optimization of intermediate bounds as a general tightening heuristic
    bound_opts['enable_opt_interm_bounds'] = True
    # Monkey-stash on function (avoids threading through many params)
    setattr(build_grid_set_from_nn, '_bound_opts', bound_opts)

    # Parse splitting args
    def _parse_int_list(opt: Optional[str]) -> Optional[List[int]]:
        if opt is None:
            return None
        s = opt.strip()
        if not s:
            return None
        return [int(x) for x in s.split(',')]

    split_dims = _parse_int_list(args.split_dims)
    splits = _parse_int_list(args.splits)

    # Auto-detect periodic dims if not explicitly provided
    try:
        _system_for_meta = instantiate_system_by_name(args.system)
    except Exception:
        _system_for_meta = None
    if split_dims is None and _system_for_meta is not None:
        periodic = getattr(_system_for_meta, 'state_periodic', None)
        if isinstance(periodic, (list, tuple)):
            split_dims = [i for i, v in enumerate(periodic) if bool(v)] or None
    if split_dims is not None and splits is None:
        splits = [8] * len(split_dims)

    # Optional comparison to pick tightest config on a sample
    if not args.no_compare and int(args.compare_sample) > 0:
        try:
            system_cmp = instantiate_system_by_name(args.system)
            meta = get_nn_input_metadata(args.nn_input_tag)
            nn_cmp = load_nn_input_by_tag(args.nn_input_tag, system_cmp, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        except Exception as e:
            print(f"Warning: comparison phase skipped due to initialization error: {e}")
            system_cmp = None
            nn_cmp = None

        if system_cmp is not None and nn_cmp is not None:
            # Build a sampled state set
            res_cfg_cmp = load_resolution_config(args.system, None, args.config)
            state_pts = _build_state_grid_points(system_cmp, res_cfg_cmp['state_resolution'])
            axes = [ax.detach().cpu() for ax in state_pts]
            shape = tuple(len(ax) for ax in axes)
            sdim = len(axes)
            meshes = torch.meshgrid(*axes, indexing='ij')
            grid_all = torch.stack(meshes, dim=-1).reshape(-1, sdim)
            Nall = int(grid_all.shape[0])
            k = min(int(args.compare_sample), Nall)
            idx = torch.randperm(Nall)[:k]
            gs = grid_all[idx]

            # Time samples
            th = float(getattr(system_cmp, 'time_horizon'))
            tr = int(max(1, int(args.compare_time_resolution)))
            t_vals = torch.linspace(0.0, th, tr)

            # Device and dtype
            tdev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            nn_cmp._module.to(tdev)
            in_dim_cmp = sdim + (0 if getattr(nn_cmp, 'time_invariant', True) else 1)
            dummy = torch.zeros(1, in_dim_cmp, device=tdev, dtype=torch.float32)

            def eval_cand(name: str, method: str, alpha: bool, beta: bool, c_split_dims: Optional[List[int]], c_splits: Optional[List[int]]) -> float:
                bo = {
                    'forward_refinement': bool(args.forward_refinement),
                    'crown_batch_size': (np.inf if int(args.crown_batch_size) <= 0 else int(args.crown_batch_size)),
                    'optimize_bound_args': {
                        'enable_alpha_crown': bool(alpha),
                        'enable_beta_crown': bool(beta),
                        'iteration': int(args.optimize_iters),
                        'lr_alpha': float(args.optimize_lr_alpha),
                        'use_float64_in_last_iteration': True,
                        'tighten_input_bounds': True,
                        'best_of_oc_and_no_oc': False,
                    },
                    'enable_opt_interm_bounds': True,
                }
                li_m = BoundedModule(nn_cmp._module, dummy, device=tdev, bound_opts=bo)
                total = 0.0
                total_count = 0
                # Prepare splitting cfg
                scfg = None
                if c_split_dims is not None and c_splits is not None:
                    scfg = list(zip(c_split_dims, c_splits))
                for t in t_vals:
                    emin, emax = system_cmp.uncertainty_limits(gs, float(t))
                    xc_all = gs.clone()
                    xl_all = xc_all + emin
                    xu_all = xc_all + emax
                    if in_dim_cmp == sdim + 1:
                        tcol = torch.full((xc_all.shape[0], 1), float(t))
                        xc_all = torch.cat([xc_all, tcol], dim=-1)
                        xl_all = torch.cat([xl_all, tcol], dim=-1)
                        xu_all = torch.cat([xu_all, tcol], dim=-1)
                    bs = int(max(1, int(args.compare_batch_size)))
                    for i in range(0, xc_all.shape[0], bs):
                        xc = xc_all[i:i+bs].to(device=tdev, dtype=torch.float32)
                        xl = xl_all[i:i+bs].to(device=tdev, dtype=torch.float32)
                        xu = xu_all[i:i+bs].to(device=tdev, dtype=torch.float32)
                        if scfg is None:
                            lb, ub = _compute_bounds_for_batch(li_m, xc, xl, xu, method=method)
                        else:
                            lb_min = None
                            ub_max = None
                            for combo in product(*[range(s) for _, s in scfg]):
                                xl_c = xl.clone(); xu_c = xu.clone()
                                for (dim, s), seg_idx in zip(scfg, combo):
                                    delta = xu[:, dim] - xl[:, dim]
                                    lower = xl[:, dim] + (float(seg_idx) / float(s)) * delta
                                    upper = xl[:, dim] + (float(seg_idx + 1) / float(s)) * delta
                                    xl_c[:, dim] = lower; xu_c[:, dim] = upper
                                lb_c, ub_c = _compute_bounds_for_batch(li_m, xc, xl_c, xu_c, method=method)
                                lb_min = lb_c if lb_min is None else torch.minimum(lb_min, lb_c)
                                ub_max = ub_c if ub_max is None else torch.maximum(ub_max, ub_c)
                            lb, ub = lb_min, ub_max
                        w = (ub - lb).float()
                        total += w.sum().item()
                        total_count += float(w.numel())
                return float(total / max(total_count, 1.0))

            # Candidate list
            candidates: List[Tuple[str, str, bool, bool, Optional[List[int]], Optional[List[int]]]] = []
            candidates.append(("base_ibp", "CROWN-IBP", True, False, None, None))
            candidates.append(("crown_alpha", "CROWN", True, False, None, None))
            if split_dims is not None:
                candidates.append(("crown_alpha_split4", "CROWN", True, False, split_dims, [4]*len(split_dims)))
                candidates.append(("crown_alpha_beta_split8", "CROWN", True, True, split_dims, [8]*len(split_dims)))

            print("Comparing candidate configurations on a sample…")
            best_name = None; best_score = float('inf'); best = None
            for name, mth, a, b, sd, sp in candidates:
                score = eval_cand(name, mth, a, b, sd, sp)
                print(f"  {name}: avg width = {score:.6g}")
                if score < best_score:
                    best_score = score; best_name = name; best = (mth, a, b, sd, sp)
            if best is not None:
                print(f"Selected config: {best_name} (avg width {best_score:.6g})")
                # Override method and bound opts
                args.method, a, b, sd, sp = best
                setattr(build_grid_set_from_nn, '_bound_opts', {
                    'forward_refinement': bool(args.forward_refinement),
                    'crown_batch_size': (np.inf if int(args.crown_batch_size) <= 0 else int(args.crown_batch_size)),
                    'optimize_bound_args': {
                        'enable_alpha_crown': bool(a),
                        'enable_beta_crown': bool(b),
                        'iteration': int(args.optimize_iters),
                        'lr_alpha': float(args.optimize_lr_alpha),
                        'use_float64_in_last_iteration': True,
                        'tighten_input_bounds': True,
                        'best_of_oc_and_no_oc': False,
                    },
                    'enable_opt_interm_bounds': True,
                })
                split_dims = sd; splits = sp

    ok = build_grid_set_from_nn(
        args.system,
        args.nn_input_tag,
        args.tag,
        args.description,
        config_path=args.config,
        time_resolution=args.time_resolution,
        batch_size=int(args.batch_size),
        method=args.method,
        device=args.device,
        force=args.force,
        split_dims=split_dims,
        splits=splits,
    )
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
