#!/usr/bin/env python3
"""
Inspect saved simulation results.

Usage:
    # List all simulation tags
    python scripts/simulation/inspect_simulation.py --list

    # Inspect a specific simulation
    python scripts/simulation/inspect_simulation.py --tag my_simulation

    # Show detailed trajectory information
    python scripts/simulation/inspect_simulation.py --tag my_simulation --verbose
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.utils.registry import list_simulation_tags as get_simulation_tags
from src.utils.table_formatter import CacheTableFormatter


def list_simulation_tags():
    """List all available simulation result tags in a formatted table."""
    sim_dir = Path('outputs') / 'simulations'
    
    tags = get_simulation_tags()
    
    if not tags:
        if not sim_dir.exists():
            print("\nNo simulation results found (outputs/simulations/ does not exist)")
        else:
            print("\nNo simulation results found in outputs/simulations/")
        return
    
    print(f"\nFound {len(tags)} simulation result(s):\n")
    
    # Define columns
    columns = {
        'Tag': 35,
        'System': 20,
        'Control': 20,
        '# Traj': 7,
        'Horizon': 10,
        'Device': 8,
        'Description': -1  # Auto-sized
    }
    
    formatter = CacheTableFormatter(columns=columns, auto_size_column='Description')
    formatter.print_header()
    formatter.print_separator()
    
    for tag in tags:
        result_path = sim_dir / tag / 'results.pkl'
        meta_path = sim_dir / tag / 'metadata.json'
        
        # Try to load metadata for quick info
        if meta_path.exists():
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                
                horizon_str = f"{meta.get('time_horizon', 'N/A'):.2f}s" if isinstance(meta.get('time_horizon'), (int, float)) else 'N/A'
                
                row_data = {
                    'Tag': tag,
                    'System': meta.get('system_name', 'N/A'),
                    'Control': meta.get('control_name', 'N/A'),
                    '# Traj': str(meta.get('n_trajectories', 'N/A')),
                    'Horizon': horizon_str,
                    'Device': meta.get('device', 'N/A'),
                    'Description': meta.get('description', ''),
                }
                
                formatter.print_row(row_data, first_line_only={'# Traj': True})
            except Exception as e:
                row_data = {
                    'Tag': tag,
                    'System': '(ERROR)',
                    'Control': '(ERROR)',
                    '# Traj': '-',
                    'Horizon': '-',
                    'Device': '-',
                    'Description': f'Error reading metadata: {e}',
                }
                formatter.print_row(row_data)
        else:
            row_data = {
                'Tag': tag,
                'System': '(no metadata)',
                'Control': '-',
                '# Traj': '-',
                'Horizon': '-',
                'Device': '-',
                'Description': '',
            }
            formatter.print_row(row_data)
    
    print()


def inspect_simulation(tag: str, verbose: bool = False):
    """Inspect a specific simulation result."""
    result_path = Path('outputs') / 'simulations' / tag / 'results.pkl'
    
    if not result_path.exists():
        print(f"\n✗ Simulation result not found: {tag}")
        print(f"  Expected path: {result_path}")
        return
    
    print(f"\nLoading simulation: {tag}")
    print(f"  Path: {result_path}")
    
    with open(result_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"\n{'='*80}")
    print("SIMULATION METADATA")
    print('='*80)
    print(f"Tag:                {data['tag']}")
    print(f"Description:        {data.get('description', 'N/A')}")
    print(f"Created:            {data['created_at']}")
    print(f"\nSYSTEM & INPUTS")
    print(f"System:             {data['system_name']}")
    print(f"Control:            {data['control_name']}")
    print(f"Disturbance:        {data['disturbance_name']}")
    print(f"Uncertainty:        {data['uncertainty_name']}")
    # Tags used (if any)
    if data.get('control_tag'):
        print(f"  Control tag:      {data['control_tag']}")
    if data.get('disturbance_tag'):
        print(f"  Disturbance tag:  {data['disturbance_tag']}")
    if data.get('uncertainty_tag'):
        print(f"  Uncertainty tag:  {data['uncertainty_tag']}")
    
    print(f"\nSIMULATION PARAMETERS")
    print(f"Time step (dt):     {data['dt']:.4f}s")
    print(f"Number of steps:    {data['steps']}")
    print(f"Time horizon:       {data['time_horizon']:.4f}s")
    print(f"Device:             {data['device']}")
    print(f"GPU requested:      {data.get('use_gpu', 'N/A')}")
    print(f"Batch size:         {data.get('batch_size', 'N/A')}")
    
    print(f"\nTRAJECTORIES")
    print(f"Number:             {data['n_trajectories']}")
    
    # Initial states
    initial_states = data['initial_states']
    print(f"Initial states:     {initial_states.shape}")
    if verbose or data['n_trajectories'] <= 10:
        for i in range(min(data['n_trajectories'], 10)):
            print(f"  [{i}]: {initial_states[i].tolist()}")
        if data['n_trajectories'] > 10:
            print(f"  ... and {data['n_trajectories'] - 10} more")
    
    # Trajectory data (stored as batched tensors)
    print(f"\nTRAJECTORY TENSORS (batched)")
    print(f"  states:           {data['states'].shape} {data['states'].dtype}")
    print(f"  controls:         {data['controls'].shape} {data['controls'].dtype}")
    print(f"  disturbances:     {data['disturbances'].shape} {data['disturbances'].dtype}")
    print(f"  uncertainties:    {data['uncertainties'].shape} {data['uncertainties'].dtype}")
    print(f"  estimated_states: {data['estimated_states'].shape} {data['estimated_states'].dtype}")
    print(f"  times:            {data['times'].shape} {data['times'].dtype}")
    
    if verbose:
        print(f"\n{'='*80}")
        print("DETAILED TRAJECTORY STATISTICS")
        print('='*80)
        
        states = data['states']  # [n_trajectories, time_steps+1, state_dim]
        controls = data['controls']  # [n_trajectories, time_steps, control_dim]
        
        for i in range(min(3, data['n_trajectories'])):
            print(f"\nTrajectory {i}:")
            traj_states = states[i]  # [time_steps+1, state_dim]
            traj_controls = controls[i]  # [time_steps, control_dim]
            
            print(f"  Initial state: {traj_states[0].tolist()}")
            print(f"  Final state:   {traj_states[-1].tolist()}")
            print(f"  State range:")
            for dim in range(traj_states.shape[-1]):
                print(f"    Dim {dim}: [{traj_states[:, dim].min():.4f}, {traj_states[:, dim].max():.4f}]")
            
            print(f"  Control range:")
            for dim in range(traj_controls.shape[-1]):
                print(f"    Dim {dim}: [{traj_controls[:, dim].min():.4f}, {traj_controls[:, dim].max():.4f}]")
        
        if data['n_trajectories'] > 3:
            print(f"\n  ... and {data['n_trajectories'] - 3} more trajectories")
    
    print(f"\n{'='*80}")
    
    # Calculate total size
    size_mb = result_path.stat().st_size / (1024**2)
    print(f"\nFile size: {size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--list', action='store_true', help='List all simulation tags')
    parser.add_argument('--tag', type=str, help='Simulation tag to inspect')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed information')
    
    args = parser.parse_args()
    
    if args.list:
        list_simulation_tags()
    elif args.tag:
        inspect_simulation(args.tag, verbose=args.verbose)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
