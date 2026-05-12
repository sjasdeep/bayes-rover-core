#!/usr/bin/env python3
"""
Inspect GridValue cache directory.

Usage:
    python scripts/grid_value/inspect_grid_value_cache.py            # list all
    python scripts/grid_value/inspect_grid_value_cache.py --tag TAG  # details for one

Lists .cache/grid_values/*.pkl with tag, size, dynamics, system, grid shape, time steps.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.cache_inspector import CacheInspector
from src.utils.cache_loaders import get_grid_value_metadata
from src.utils.table_formatter import format_shape


def list_cached():
    columns = {
        'Tag': 40,
        'Size(MB)': 9,
        'Dynamics': 25,
        'System': 20,
        'Shape': 20,
        'Steps': 6,
        'Description': -1
    }
    
    column_extractors = {
        'Tag': lambda path, meta: path.stem,
        'Size(MB)': lambda path, meta: f"{path.stat().st_size / (1024**2):.2f}",
        'Dynamics': lambda path, meta: meta.get('dynamics_name', 'unknown'),
        'System': lambda path, meta: meta.get('system_name', 'unknown'),
        'Shape': lambda path, meta: format_shape(meta.get('grid_shape', [])),
        'Steps': lambda path, meta: str(meta['time_steps']) if meta.get('time_steps') is not None else '-',
        'Description': lambda path, meta: meta.get('description', ''),
    }
    
    inspector = CacheInspector(
        cache_subdir='grid_values',
        metadata_getter=get_grid_value_metadata,
        columns=columns,
        column_extractors=column_extractors,
        auto_size_column='Description'
    )
    
    inspector.print_table()


def inspect_one(tag: str):
    cache_path = Path('.cache') / 'grid_values' / f'{tag}.pkl'
    if not cache_path.exists():
        print(f"\n✗ GridValue cache not found: {tag}")
        print(f"  Expected path: {cache_path}")
        return
    
    print(f"\nGridValue Cache: {tag}")
    print(f"Path: {cache_path}")
    print(f"Size: {cache_path.stat().st_size / (1024**2):.2f} MB")
    
    try:
        meta = get_grid_value_metadata(tag)
        print(f"\n{'='*80}")
        print("METADATA")
        print('='*80)
        print(f"Dynamics:       {meta['dynamics_name']}")
        print(f"System:         {meta['system_name']}")
        print(f"Description:    {meta.get('description', 'N/A')}")
        print(f"\nGRID")
        print(f"Shape:          {format_shape(meta['grid_shape'])}")
        print(f"State dim:      {meta.get('state_dim', 'N/A')}")
        print(f"\nTIME")
        print(f"Steps:          {meta.get('time_steps', 'N/A')}")
        print(f"Horizon:        {meta.get('time_horizon', 'N/A')}")
        
        if 'control_grid_set_tag' in meta and meta['control_grid_set_tag']:
            print(f"\nCONTROL")
            print(f"GridSet tag:    {meta['control_grid_set_tag']}")
        
        if 'disturbance_grid_set_tag' in meta and meta['disturbance_grid_set_tag']:
            print(f"\nDISTURBANCE")
            print(f"GridSet tag:    {meta['disturbance_grid_set_tag']}")
        
        if 'uncertainty_grid_set_tag' in meta and meta['uncertainty_grid_set_tag']:
            print(f"\nUNCERTAINTY")
            print(f"GridSet tag:    {meta['uncertainty_grid_set_tag']}")
        
        print(f"\n{'='*80}")
    except Exception as e:
        print(f"\n✗ Error reading metadata: {e}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--tag', type=str, help='Show detailed info for specific tag')
    
    args = parser.parse_args()
    
    if args.tag:
        inspect_one(args.tag)
    else:
        list_cached()


if __name__ == '__main__':
    main()
