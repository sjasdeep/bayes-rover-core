"""
High-level cache inspection utilities.

Provides a base class for implementing cache inspection scripts with
consistent formatting and error handling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.utils.table_formatter import CacheTableFormatter, format_shape

__all__ = ["CacheInspector"]


class CacheInspector:
    """Base class for cache directory inspection with formatted table output."""
    
    def __init__(
        self,
        cache_subdir: str,
        metadata_getter: Callable[[str], Dict[str, Any]],
        columns: Dict[str, int],
        column_extractors: Dict[str, Callable[[Path, Dict[str, Any]], Any]],
        auto_size_column: Optional[str] = 'Description'
    ):
        """
        Initialize cache inspector.
        
        Args:
            cache_subdir: Subdirectory under .cache/ (e.g., 'grid_inputs', 'grid_sets').
            metadata_getter: Function that takes a tag and returns metadata dict.
            columns: Dictionary mapping column names to their display widths.
            column_extractors: Dictionary mapping column names to functions that extract
                             the column value from (file_path, metadata).
            auto_size_column: Name of column to auto-size to terminal (typically 'Description').
        """
        self.cache_dir = Path('.cache') / cache_subdir
        self.get_metadata = metadata_getter
        self.columns = columns
        self.column_extractors = column_extractors
        self.formatter = CacheTableFormatter(
            columns=columns,
            auto_size_column=auto_size_column
        )
    
    def list_files(self) -> List[Path]:
        """List all .pkl files in cache directory, sorted by name."""
        if not self.cache_dir.exists():
            return []
        return sorted(self.cache_dir.glob('*.pkl'))
    
    def extract_row_data(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract row data from cache file with error handling.
        
        Args:
            file_path: Path to cache file.
            
        Returns:
            Dictionary mapping column names to their values.
        """
        tag = file_path.stem
        size_mb = file_path.stat().st_size / (1024**2)
        
        # Try to load metadata
        try:
            meta = self.get_metadata(tag)
            error = None
        except Exception as e:
            meta = {}
            error = str(e)
        
        # Extract column values
        row = {}
        for col_name, extractor in self.column_extractors.items():
            try:
                if error and col_name != 'Tag' and col_name != 'Size(MB)':
                    # Show error in description or first text column
                    if col_name == 'Description':
                        row[col_name] = f'ERROR: {error}'
                    else:
                        row[col_name] = '(ERROR)'
                else:
                    row[col_name] = extractor(file_path, meta)
            except Exception as e:
                row[col_name] = f'ERROR: {e}'
        
        return row
    
    def print_table(self):
        """Print formatted table of all cache files."""
        files = self.list_files()
        
        if not files:
            print(f"No cache files found in {self.cache_dir}")
            return
        
        print(f"\nFound {len(files)} cache file(s) in {self.cache_dir}:\n")
        
        # Print header
        self.formatter.print_header()
        self.formatter.print_separator()
        
        # Print rows
        for file_path in files:
            row_data = self.extract_row_data(file_path)
            # Mark Size(MB) and similar numeric columns as first-line-only
            first_line_only = {
                name: True for name in self.columns.keys()
                if any(x in name for x in ['Size', 'MB', 'Steps', '#'])
            }
            self.formatter.print_row(row_data, first_line_only=first_line_only)
        
        print()  # Empty line at end
