#!/usr/bin/env python3
"""
List all tagged GridInput caches with basic info.

Usage:
    python scripts/grid_input/inspect_grid_input_cache.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.cache_inspector import CacheInspector
from src.utils.cache_loaders import get_grid_input_metadata
from src.utils.table_formatter import format_shape


def main():
    # Define columns with their display widths
    columns = {
        'Tag': 40,
        'System': 20,
        'Input': 30,
        'Size(MB)': 9,
        'Shape': 20,
        'Description': -1  # Auto-sized to terminal width
    }
    
    # Define how to extract each column from file path and metadata
    column_extractors = {
        'Tag': lambda path, meta: path.stem,
        'System': lambda path, meta: meta.get('system_name', 'unknown'),
        'Input': lambda path, meta: meta.get('input_name', 'unknown'),
        'Size(MB)': lambda path, meta: f"{path.stat().st_size / (1024**2):.2f}",
        'Shape': lambda path, meta: format_shape(meta.get('grid_shape', [])),
        'Description': lambda path, meta: meta.get('description', ''),
    }
    
    # Create inspector and print table
    inspector = CacheInspector(
        cache_subdir='grid_inputs',
        metadata_getter=get_grid_input_metadata,
        columns=columns,
        column_extractors=column_extractors,
        auto_size_column='Description'
    )
    
    inspector.print_table()


if __name__ == '__main__':
    main()
