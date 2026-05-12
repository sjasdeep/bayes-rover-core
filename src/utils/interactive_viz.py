#!/usr/bin/env python3
"""
Application-agnostic interactive visualization utility.

This module provides tools for creating interactive visualizations with sliders
to explore multi-dimensional data. It supports both Jupyter notebook environments
(using ipywidgets) and standalone matplotlib windows (using matplotlib.widgets).

The main class, InteractiveVisualizer, accepts a callback function that generates
matplotlib figures based on slider parameters, handling all the widget creation
and event management automatically.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple, Optional, Any
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

__all__ = ["SliderSpec", "InteractiveVisualizer"]


class SliderSpec:
    """
    Specification for a single slider parameter.
    
    Attributes:
        name: Display name for the slider
        min_val: Minimum value
        max_val: Maximum value
        initial_val: Initial value (default: midpoint)
        step: Step size (default: auto-computed based on range)
        description: Optional description text
    """
    
    def __init__(
        self,
        name: str,
        min_val: float,
        max_val: float,
        initial_val: Optional[float] = None,
        step: Optional[float] = None,
        description: Optional[str] = None
    ):
        self.name = name
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        
        if initial_val is None:
            self.initial_val = (self.min_val + self.max_val) / 2
        else:
            self.initial_val = float(initial_val)
            
        if step is None:
            # Auto-compute reasonable step size
            range_val = self.max_val - self.min_val
            if range_val > 100:
                self.step = range_val / 100
            elif range_val > 10:
                self.step = range_val / 50
            else:
                self.step = range_val / 20
        else:
            self.step = float(step)
            
        self.description = description or name
    
    def __repr__(self):
        return f"SliderSpec(name='{self.name}', range=[{self.min_val}, {self.max_val}], initial={self.initial_val})"


class InteractiveVisualizer:
    """
    Interactive visualization with slider controls.
    
    This class creates an interactive visualization where the user can adjust
    sliders to explore different slices or time steps of multi-dimensional data.
    
    The visualizer is backend-agnostic: it will automatically use ipywidgets if
    available (for Jupyter notebooks) or fall back to matplotlib.widgets for
    standalone windows.
    
    Usage:
        # Define slider specifications
        sliders = [
            SliderSpec('time', 0, 10, initial_val=0, step=0.1, description='Time (s)'),
            SliderSpec('dim_2', -5, 5, initial_val=0, description='Fixed dimension 2')
        ]
        
        # Define update function that generates figure based on slider values
        def update_func(time, dim_2):
            fig, ax = plt.subplots()
            # ... generate plot based on parameters ...
            return fig
        
        # Create and show interactive visualization
        viz = InteractiveVisualizer(sliders, update_func)
        viz.show()
    """
    
    def __init__(
        self,
        sliders: List[SliderSpec],
        update_function: Callable[..., plt.Figure],
        title: Optional[str] = None,
        backend: Optional[str] = None,
        direct_plot: bool = False,
        figsize: tuple = (10, 8)
    ):
        """
        Initialize the interactive visualizer.
        
        Args:
            sliders: List of SliderSpec objects defining the interactive parameters
            update_function: Callable that takes slider values as arguments and either:
                           - Returns a matplotlib Figure (if direct_plot=False), OR
                           - Accepts an 'ax' keyword argument and plots directly (if direct_plot=True)
            title: Optional title for the visualization
            backend: Force specific backend ('ipywidgets', 'matplotlib', or None for auto)
            direct_plot: If True, update_function should accept 'ax' parameter and plot directly
            figsize: Tuple (width, height) for figure size in inches
        """
        self.sliders = sliders
        self.update_function = update_function
        self.direct_plot = direct_plot
        self.title = title or "Interactive Visualization"
        self.figsize = figsize
        
        # Detect or use specified backend
        if backend is None:
            self.backend = self._detect_backend()
        else:
            self.backend = backend
            
        # State tracking
        self._widgets = {}
        self._current_values = {s.name: s.initial_val for s in sliders}
        self._fig = None
        self._ax = None
        
    def _detect_backend(self) -> str:
        """
        Detect which backend to use based on environment.
        
        Returns:
            'ipywidgets' if in Jupyter with ipywidgets available, else 'matplotlib'
        """
        try:
            # Check if we're in a Jupyter environment
            from IPython import get_ipython
            ipython = get_ipython()
            if ipython is not None and 'IPKernelApp' in ipython.config:
                # Try to import ipywidgets
                try:
                    import ipywidgets
                    return 'ipywidgets'
                except ImportError:
                    warnings.warn("Jupyter detected but ipywidgets not available. Falling back to matplotlib widgets.")
                    return 'matplotlib'
        except ImportError:
            pass
        
        return 'matplotlib'
    
    def show(self):
        """
        Display the interactive visualization.
        
        This will create the appropriate widget interface based on the detected backend
        and display the initial visualization.
        """
        if self.backend == 'ipywidgets':
            self._show_ipywidgets()
        else:
            self._show_matplotlib()
    
    def _show_ipywidgets(self):
        """Create and display ipywidgets-based interactive visualization."""
        try:
            import ipywidgets as widgets
            from IPython.display import display
        except ImportError:
            raise ImportError("ipywidgets backend requested but ipywidgets not installed. "
                            "Install with: pip install ipywidgets")
        
        # Use interactive backend for Jupyter
        matplotlib.use('module://matplotlib_inline.backend_inline')
        
        # Create slider widgets
        slider_widgets = []
        for spec in self.sliders:
            slider = widgets.FloatSlider(
                value=spec.initial_val,
                min=spec.min_val,
                max=spec.max_val,
                step=spec.step,
                description=spec.description,
                continuous_update=False,  # Update only on release for better performance
                readout=True,
                readout_format='.3f',
                layout=widgets.Layout(width='60%')
            )
            self._widgets[spec.name] = slider
            slider_widgets.append(slider)
        
        # Create interactive output
        def update_wrapper(**kwargs):
            # Close previous figure to avoid accumulation
            plt.close('all')
            
            # Update current values
            for name, value in kwargs.items():
                self._current_values[name] = value
            
            # Call user's update function with slider values in correct order
            args = [self._current_values[s.name] for s in self.sliders]
            fig = self.update_function(*args)
            
            # Display the figure
            plt.show()
        
        # Create interactive widget
        interactive_widget = widgets.interactive_output(
            update_wrapper,
            {s.name: self._widgets[s.name] for s in self.sliders}
        )
        
        # Layout
        ui = widgets.VBox([
            widgets.HTML(f"<h3>{self.title}</h3>"),
            *slider_widgets,
            interactive_widget
        ])
        
        # Display
        display(ui)
        
        # Initial update
        update_wrapper(**{s.name: s.initial_val for s in self.sliders})
    
    def _show_matplotlib(self):
        """Create and display matplotlib.widgets-based interactive visualization."""
        from matplotlib.widgets import Slider
        
        # Use interactive backend
        original_backend = matplotlib.get_backend()
        if 'Agg' in original_backend:
            # Switch from non-interactive to interactive backend
            try:
                matplotlib.use('TkAgg')
            except:
                try:
                    matplotlib.use('Qt5Agg')
                except:
                    warnings.warn("Could not switch to interactive backend. "
                                "Visualization may not be interactive.")
        
        # Calculate layout dimensions
        n_sliders = len(self.sliders)
        slider_height = 0.03
        slider_spacing = 0.01
        bottom_margin = 0.05
        top_margin = 0.02
        plot_xlabel_space = 0.05  # Extra space for x-axis label between plot and sliders
        top_title_space = 0.10    # Extra space at top for figure title and subplot titles
        
        total_slider_height = n_sliders * (slider_height + slider_spacing) + bottom_margin + top_margin
        plot_bottom = total_slider_height + plot_xlabel_space
        plot_height = 1.0 - plot_bottom - top_title_space
        
        # Create figure with space for sliders at bottom
        self._fig = plt.figure(figsize=self.figsize)
        # Position title near top, leaving room for subplot titles below it
        self._fig.suptitle(self.title, fontsize=14, y=0.97)
        
        # Create axis for the main plot
        self._ax = self._fig.add_axes([0.1, plot_bottom, 0.85, plot_height])
        
        # Create slider axes and widgets
        slider_widgets = []
        for i, spec in enumerate(self.sliders):
            # Position from bottom up
            slider_bottom = bottom_margin + i * (slider_height + slider_spacing)
            ax_slider = self._fig.add_axes([0.15, slider_bottom, 0.7, slider_height])
            
            slider = Slider(
                ax_slider,
                spec.description,
                spec.min_val,
                spec.max_val,
                valinit=spec.initial_val,
                valstep=spec.step
            )
            
            self._widgets[spec.name] = slider
            slider_widgets.append(slider)
        
        # Store references to plot elements for efficient updates
        self._plot_elements = {}
        self._colorbar = None
        
        # Define update callback
        def update_matplotlib(val):
            # Update current values from all sliders
            for spec in self.sliders:
                self._current_values[spec.name] = self._widgets[spec.name].val
            
            # Clear the main axis and colorbar (only if axis still has a figure)
            if self._ax.figure is not None:
                self._ax.clear()
            if self._colorbar is not None:
                self._colorbar.remove()
                self._colorbar = None
            
            # Call user's update function with slider values in correct order
            args = [self._current_values[s.name] for s in self.sliders]
            
            if self.direct_plot:
                # Direct plotting mode: pass axis to update function
                self.update_function(*args, ax=self._ax)
            else:
                # Legacy mode: copy from temporary figure
                temp_fig = self.update_function(*args)
                
                # Extract data and recreate plot in our axis
                if temp_fig is not None and len(temp_fig.get_axes()) > 0:
                    # Handle the first user axis (skip slider axes if any)
                    source_axes = [ax for ax in temp_fig.get_axes() if ax.get_label() != '<colorbar>']
                    if len(source_axes) == 0:
                        source_axes = temp_fig.get_axes()
                source_ax = source_axes[0]
                
                # Handle filled contours (QuadContourSet from contourf)
                from matplotlib.contour import QuadContourSet
                from matplotlib.collections import PolyCollection, PathCollection, LineCollection
                
                for collection in source_ax.collections:
                    if isinstance(collection, QuadContourSet):
                        # Copy contourf by extracting paths and using original colors
                        paths = collection.get_paths()
                        facecolors = collection.get_facecolors()
                        
                        # For contourf, we don't want edge lines between polygons
                        new_collection = PolyCollection(
                            [p.vertices for p in paths],
                            facecolors=facecolors,
                            edgecolors='none',  # No edges for smooth contours
                            antialiaseds=True,
                            rasterized=False
                        )
                        self._ax.add_collection(new_collection)
                        self._ax.autoscale()
                        
                        # For colorbar, we need to create a ScalarMappable with the original norm/cmap
                        from matplotlib.cm import ScalarMappable
                        sm = ScalarMappable(cmap=collection.get_cmap(), norm=collection.norm)
                        sm.set_array(collection.get_array())
                        
                        # Add colorbar
                        if self._colorbar is None:
                            self._colorbar = plt.colorbar(sm, ax=self._ax)
                    
                    elif isinstance(collection, PolyCollection):
                        # Handle other PolyCollections
                        paths = collection.get_paths()
                        facecolors = collection.get_facecolors()
                        edgecolors = collection.get_edgecolors()
                        linewidths = collection.get_linewidths()
                        
                        new_collection = PolyCollection(
                            [p.vertices for p in paths],
                            facecolors=facecolors,
                            edgecolors=edgecolors,
                            linewidths=linewidths
                        )
                        self._ax.add_collection(new_collection)
                        self._ax.autoscale()
                    
                    elif isinstance(collection, (PathCollection, LineCollection)):
                        # Handle scatter and line collections
                        if isinstance(collection, PathCollection):
                            offsets = collection.get_offsets()
                            if len(offsets) > 0:
                                self._ax.scatter(
                                    offsets[:, 0], offsets[:, 1],
                                    c=collection.get_facecolors(),
                                    s=collection.get_sizes() if hasattr(collection, 'get_sizes') else 20
                                )
                
                # Handle images
                for img in source_ax.get_images():
                    self._ax.imshow(
                        img.get_array(),
                        extent=img.get_extent(),
                        aspect=img.get_aspect(),
                        cmap=img.get_cmap(),
                        interpolation=img.get_interpolation(),
                        origin=img.origin,
                        norm=img.norm
                    )
                
                # Handle line plots
                for line in source_ax.get_lines():
                    self._ax.plot(
                        line.get_xdata(),
                        line.get_ydata(),
                        color=line.get_color(),
                        linewidth=line.get_linewidth(),
                        linestyle=line.get_linestyle(),
                        marker=line.get_marker(),
                        label=line.get_label()
                    )
                
                # Copy axis properties
                self._ax.set_xlabel(source_ax.get_xlabel())
                self._ax.set_ylabel(source_ax.get_ylabel())
                self._ax.set_title(source_ax.get_title())
                self._ax.set_xlim(source_ax.get_xlim())
                self._ax.set_ylim(source_ax.get_ylim())
                
                # Copy grid settings
                try:
                    # Check if grid is visible on the source
                    if source_ax.xaxis.get_gridlines()[0].get_visible():
                        self._ax.grid(True, alpha=0.3)
                except (AttributeError, IndexError, KeyError):
                    pass
                
                    # Close temporary figure
                    plt.close(temp_fig)
            
            # Redraw
            self._fig.canvas.draw_idle()
        
        # Connect sliders to update function
        for slider in slider_widgets:
            slider.on_changed(update_matplotlib)
        
        # Initial update
        update_matplotlib(None)
        
        # Show the plot
        plt.show()

    
    def get_current_values(self) -> Dict[str, float]:
        """
        Get the current values of all sliders.
        
        Returns:
            Dictionary mapping slider names to current values
        """
        return self._current_values.copy()
    
    def save_current(self, filepath: str, **kwargs):
        """
        Save the current visualization to a file.
        
        Args:
            filepath: Output file path
            **kwargs: Additional arguments passed to matplotlib's savefig
        """
        # Generate figure with current slider values
        args = [self._current_values[s.name] for s in self.sliders]
        fig = self.update_function(*args)
        
        if fig is not None:
            fig.savefig(filepath, **kwargs)
            plt.close(fig)
            print(f"✓ Saved to {filepath}")


def create_dimension_sliders(
    state_dim: int,
    state_limits: np.ndarray,
    state_labels: Optional[List[str]] = None,
    exclude_dims: Optional[List[int]] = None,
    initial_values: Optional[Dict[int, float]] = None
) -> List[SliderSpec]:
    """
    Helper function to create sliders for state dimensions.
    
    Args:
        state_dim: Number of state dimensions
        state_limits: Array of shape [2, state_dim] with [min, max] limits
        state_labels: Optional list of labels for each dimension
        exclude_dims: Optional list of dimension indices to exclude from sliders
        initial_values: Optional dict mapping dimension index to initial value
    
    Returns:
        List of SliderSpec objects for each non-excluded dimension
    """
    exclude_dims = exclude_dims or []
    initial_values = initial_values or {}
    state_labels = state_labels or [f"dim_{i}" for i in range(state_dim)]
    
    sliders = []
    for i in range(state_dim):
        if i in exclude_dims:
            continue
        
        min_val = float(state_limits[0, i])
        max_val = float(state_limits[1, i])
        initial = initial_values.get(i, (min_val + max_val) / 2)
        
        slider = SliderSpec(
            name=f"dim_{i}",
            min_val=min_val,
            max_val=max_val,
            initial_val=initial,
            description=state_labels[i]
        )
        sliders.append(slider)
    
    return sliders


def create_time_slider(
    time_points: np.ndarray,
    description: str = "Time"
) -> SliderSpec:
    """
    Helper function to create a time slider.
    
    Args:
        time_points: Array of time values
        description: Description for the slider
    
    Returns:
        SliderSpec for time selection
    """
    min_time = float(np.min(time_points))
    max_time = float(np.max(time_points))
    
    # Use step size that allows hitting most/all time points
    if len(time_points) > 1:
        step = float(np.min(np.diff(np.sort(time_points))))
    else:
        step = (max_time - min_time) / 100
    
    return SliderSpec(
        name="time",
        min_val=min_time,
        max_val=max_time,
        initial_val=min_time,
        step=step,
        description=description
    )


def create_index_slider(
    max_index: int,
    description: str = "Index"
) -> SliderSpec:
    """
    Helper function to create an integer index slider.
    
    Args:
        max_index: Maximum index value (inclusive)
        description: Description for the slider
    
    Returns:
        SliderSpec for index selection
    """
    return SliderSpec(
        name="index",
        min_val=0,
        max_val=max_index,
        initial_val=0,
        step=1,
        description=description
    )
