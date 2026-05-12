"""
Unified table formatter for cache inspection scripts.

Provides utilities for printing nicely formatted tables with:
- Automatic column wrapping
- Multi-line cell support
- Dynamic terminal width adjustment
- Consistent formatting across all inspect scripts
"""

from __future__ import annotations

import shutil
import textwrap
from typing import Any, Dict, List, Optional

__all__ = ["CacheTableFormatter", "format_shape"]


class CacheTableFormatter:
    """Unified table formatter for cache inspection scripts with multi-line cell support."""
    
    def __init__(
        self,
        columns: Dict[str, int],
        auto_size_column: Optional[str] = None,
        min_auto_width: int = 20,
        max_auto_width: int = 80
    ):
        """
        Initialize table formatter.
        
        Args:
            columns: Dictionary mapping column names to their widths.
                     Use positive integers for fixed widths.
            auto_size_column: Name of column to auto-size to terminal width (typically 'Description').
            min_auto_width: Minimum width for auto-sized column.
            max_auto_width: Maximum width for auto-sized column.
        """
        self.columns = columns
        self.column_names = list(columns.keys())
        self.auto_size_column = auto_size_column
        self.min_auto_width = min_auto_width
        self.max_auto_width = max_auto_width
        
        # Calculate auto-sized column width if specified
        if auto_size_column and auto_size_column in columns:
            term_width = shutil.get_terminal_size(fallback=(160, 40)).columns
            # Calculate width used by fixed columns
            fixed_width = sum(w for name, w in columns.items() if name != auto_size_column)
            # Add spacing between columns (n_columns - 1 spaces, plus 2 for size column)
            spacing = len(columns) + 2
            available = term_width - fixed_width - spacing
            self.columns[auto_size_column] = max(min_auto_width, min(max_auto_width, available))
    
    @staticmethod
    def wrap_column(text: Any, width: int) -> List[str]:
        """
        Wrap text to fit within column width.
        
        Args:
            text: Text to wrap (converted to string).
            width: Maximum width for wrapped lines.
            
        Returns:
            List of wrapped lines.
        """
        s = '' if text is None else str(text)
        if not s:
            return ['']
        return textwrap.wrap(s, width=width, break_long_words=True, break_on_hyphens=False) or ['']
    
    def print_header(self):
        """Print table header with column names."""
        # Build format string
        fmt_parts = []
        for name in self.column_names:
            width = self.columns[name]
            # Right-align if column name suggests it's numeric (Size, MB, etc.)
            if any(x in name for x in ['Size', 'MB', 'Steps', '#']):
                fmt_parts.append(f"{{:>{width}}}")
            else:
                fmt_parts.append(f"{{:<{width}}}")
        
        # Determine spacing
        spacing_parts = []
        for i, name in enumerate(self.column_names):
            if i > 0:
                # Add extra space before Size/numeric columns
                if any(x in name for x in ['Size', 'MB']):
                    spacing_parts.append("  ")
                else:
                    spacing_parts.append(" ")
            spacing_parts.append(fmt_parts[i])
        
        format_string = "".join(spacing_parts)
        print(format_string.format(*self.column_names))
    
    def print_separator(self):
        """Print separator line under header."""
        total_width = sum(self.columns.values())
        # Add spacing between columns
        spacing = len(self.columns) + 2  # account for extra spaces
        print('-' * (total_width + spacing))
    
    def print_row(self, values: Dict[str, Any], first_line_only: Optional[Dict[str, Any]] = None):
        """
        Print table row with multi-line cell wrapping support.
        
        Args:
            values: Dictionary mapping column names to their values.
            first_line_only: Optional dict of column names whose values should only appear on first line
                           (e.g., {'Size(MB)': True} for numeric values).
        """
        if first_line_only is None:
            first_line_only = {}
        
        # Wrap all columns
        wrapped_columns = {}
        for name in self.column_names:
            value = values.get(name, '')
            wrapped_columns[name] = self.wrap_column(value, self.columns[name])
        
        # Determine max lines needed
        max_lines = max(len(lines) for lines in wrapped_columns.values())
        
        # Pad all columns to same length
        for name in self.column_names:
            wrapped_columns[name] += [''] * (max_lines - len(wrapped_columns[name]))
        
        # Build format string (same as header)
        fmt_parts = []
        for name in self.column_names:
            width = self.columns[name]
            if any(x in name for x in ['Size', 'MB', 'Steps', '#']):
                fmt_parts.append(f"{{:>{width}}}")
            else:
                fmt_parts.append(f"{{:<{width}}}")
        
        spacing_parts = []
        for i, name in enumerate(self.column_names):
            if i > 0:
                if any(x in name for x in ['Size', 'MB']):
                    spacing_parts.append("  ")
                else:
                    spacing_parts.append(" ")
            spacing_parts.append(fmt_parts[i])
        
        format_string = "".join(spacing_parts)
        
        # Print each line
        for line_idx in range(max_lines):
            line_values = []
            for name in self.column_names:
                if line_idx == 0:
                    line_values.append(wrapped_columns[name][line_idx])
                elif name in first_line_only:
                    line_values.append('')  # Empty for first-line-only columns after first line
                else:
                    line_values.append(wrapped_columns[name][line_idx])
            
            print(format_string.format(*line_values))


def format_shape(shape: Any) -> str:
    """
    Format a shape tuple/list into a compact string like '100x100x50'.
    
    Args:
        shape: Shape as tuple, list, or other representation.
        
    Returns:
        Formatted shape string.
    """
    try:
        if isinstance(shape, (list, tuple)) and all(isinstance(x, (int, float)) for x in shape):
            return 'x'.join(str(int(x)) for x in shape)
        else:
            return str(shape)
    except Exception:
        return str(shape)
