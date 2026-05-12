#!/usr/bin/env python3
"""
List all tagged NNInput caches with basic info.

Usage:
    python scripts/nn_input/inspect_nn_input_cache.py
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.cache_loaders import get_nn_input_metadata
from src.utils.table_formatter import CacheTableFormatter


def list_nn_inputs():
    cache_dir = Path('.cache') / 'nn_inputs'
    if not cache_dir.exists():
        return []
    # Use .pth as the primary payload file for sizing; metadata is in .meta.json
    return sorted(cache_dir.glob('*.pth'))


def main():
    columns = {
        'Tag': 40,
        'System': 20,
        'Input': 26,
        'Type': 12,
        'Size(MB)': 9,
        'Sizes': 22,
        'Description': -1,
    }
    fmt = CacheTableFormatter(columns=columns, auto_size_column='Description')

    files = list_nn_inputs()
    if not files:
        print(f"No cache files found in .cache/nn_inputs")
        return

    print(f"\nFound {len(files)} cache file(s) in .cache/nn_inputs:\n")
    fmt.print_header()
    fmt.print_separator()

    for f in files:
        tag = f.stem
        size_mb = f.stat().st_size / (1024**2)
        try:
            meta = get_nn_input_metadata(tag)
        except Exception as e:
            meta = {
                'system_name': '(ERROR)',
                'input_name': '(ERROR)',
                'input_type': '(ERROR)',
                'sizes': None,
                'description': f'ERROR: {e}',
            }

        system = meta.get('system_name') or 'unknown'
        input_name = meta.get('input_name') or meta.get('input_class') or 'NNInput'
        input_type = meta.get('input_type') or 'unknown'
        sizes = meta.get('sizes')
        sizes_str = 'x'.join(str(s) for s in sizes) if isinstance(sizes, (list, tuple)) else str(sizes)
        description = meta.get('description', '')

        row = {
            'Tag': tag,
            'System': system,
            'Input': input_name,
            'Type': input_type,
            'Size(MB)': f"{size_mb:.2f}",
            'Sizes': sizes_str,
            'Description': description,
        }
        fmt.print_row(row, first_line_only={'Size(MB)': True})

    print()


if __name__ == '__main__':
    main()
