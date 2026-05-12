#!/usr/bin/env python3
"""
Inspect Monte Carlo value caches under .cache/monte_carlo_values.

Usage:
  # List tags
  python scripts/monte_carlo_value/inspect_cache.py --list

  # Inspect specific tag
  python scripts/monte_carlo_value/inspect_cache.py --tag my_mc
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

from src.utils.registry import list_monte_carlo_value_tags
from src.utils.table_formatter import CacheTableFormatter


def list_tags():
    tags = list_monte_carlo_value_tags()
    d = Path('.cache') / 'monte_carlo_values'
    if not tags:
        if not d.exists():
            print("\nNo Monte Carlo caches found (.cache/monte_carlo_values does not exist)")
        else:
            print("\nNo Monte Carlo caches found in .cache/monte_carlo_values/")
        return

    print(f"\nFound {len(tags)} Monte Carlo cache(s):\n")
    columns = {
        'Tag': 35,
        'System': 18,
        'Control': 20,
        'Grid': 12,
        '#Snaps': 7,
        'Horizon': 10,
        'Description': -1,
    }
    fmt = CacheTableFormatter(columns=columns, auto_size_column='Description')
    fmt.print_header(); fmt.print_separator()
    for tag in tags:
        meta_path = d / f"{tag}.meta.json"
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            grid = meta.get('shape2d', ['?', '?'])
            row = {
                'Tag': tag,
                'System': meta.get('system_name', 'N/A'),
                'Control': meta.get('control_name', 'N/A'),
                'Grid': f"{grid[0]}x{grid[1]}",
                '#Snaps': str(meta.get('snapshot_count', '0')),
                'Horizon': f"{meta.get('time_horizon', 0.0):.2f}s" if isinstance(meta.get('time_horizon'), (int, float)) else 'N/A',
                'Description': meta.get('description', ''),
            }
        else:
            row = {
                'Tag': tag,
                'System': '(no meta)',
                'Control': '-',
                'Grid': '-',
                '#Snaps': '-',
                'Horizon': '-',
                'Description': '',
            }
        fmt.print_row(row)
    print()


def inspect_tag(tag: str):
    path = Path('.cache') / 'monte_carlo_values' / f'{tag}.pkl'
    if not path.exists():
        print(f"\n✗ Cache not found: {tag}")
        print(f"  Expected at: {path}")
        return
    with open(path, 'rb') as f:
        data = pickle.load(f)
    axes = data['axes']
    shape2d = data['meta']['shape2d']
    print(f"\nTag:         {tag}")
    print(f"System:      {data.get('system_name')}")
    print(f"Control:     {data.get('control_name')}")
    print(f"Grid:        {shape2d[0]} x {shape2d[1]}")
    print(f"Vary dims:   {data['meta'].get('vary_dims')}")
    print(f"Fixed dims:  {data['meta'].get('fixed')}")
    print(f"Time horizon:{data.get('time_horizon'):.3f}s  |  time_resolution={data['meta'].get('time_resolution')}")
    print(f"Snapshots:   {len(data.get('snapshots', []))}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--tag', type=str)
    args = ap.parse_args()
    if args.list:
        list_tags()
    elif args.tag:
        inspect_tag(args.tag)
    else:
        ap.print_help()
