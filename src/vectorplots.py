import numpy as np
import plotly.graph_objects as go

from typing import Iterable, NamedTuple, Optional, Literal

from src.vector import Vector


class VectorTraces(NamedTuple):
    """
    Pair of pure `plotly.graph_objects.Trace` that should form a vector,
    i.e. line and tip.
    Useful since plotly itself has no native support for vector-stylized
    traces.
    """
    line: go.Trace
    tip: go.Trace


def create_label(instance_ID: str, base_label: str, terminal_label: str) -> str:
    """
    Create text label to display for a direction vector inside the plot.

    Parameters
    ----------

    instance_ID : str
        Dataset instance ID as string.

    base_label : str
        Label for the vector base coordinates.

    terminal_label : str
        Label for the vector terminal or tip coordinates.
    
    Parameters
    ----------

    label : str
    """
    # gerber style : GID_21
    if '_' in instance_ID and instance_ID.endswith('CT'):
        # wimmerstyle F05_CT
        ID = instance_ID.split('_')[0]
    elif '_' in instance_ID and instance_ID.endswith('UE'):
        # sieberstyle delta_4_UE
        ID = instance_ID.split('_')[0]
    elif '_' in instance_ID and instance_ID.startswith('GID'):
        ID = ''.join((instance_ID.split('_')))
    else:
        # exvivo style : 44
        # invivo style : 05
        ID = instance_ID
    return f'{ID}::{base_label}->{terminal_label}'


def build_line_trace(start_coordinates: Iterable[int],
                     end_coordinates: Iterable[int],
                     **kwargs) -> go.Trace:
    """
    Build a line trace connecting the provided start and end coordinates in
    3D Euclidean space coordinates. 
    Additional kwargs are passed through to `plotly.graph_objects.Scatter3d`
    """
    coordinates = {
        cname : (cs, ce)
        for cname, cs, ce in zip(('x', 'y', 'z'), start_coordinates, end_coordinates)
    }
    # defaults = {'mode' : 'lines',
    #             'line' : {'color' : 'blue', 'width' : 3},
    #             'name' : 'ConnLine'}
    # kwargs = {**defaults, **kwargs}
    trace = go.Scatter3d(
        **coordinates,
        **kwargs
    )
    return trace


def build_cone_trace(base_coordinates: Iterable[int],
                     terminal_coordinates: Iterable[int],
                     start_fraction: float,
                     tip_fraction: float,
                     **kwargs) -> go.Trace:
    """
    Build a vector tip 'cone' for the direction vector pointing from the
    base coordinates to the terminal coordinates.
    `start_fraction` and `tip_fraction` control the relative size of the
    tip. Sane starting values: 
        - `start_fraction = 0.98`
        - `tip_fraction = 0.1`

    Additional kwargs are passed through to `plotly.graph_objects.Cone` 
    """
    distances = np.array(terminal_coordinates) - np.array(base_coordinates)
    base_coordinates = base_coordinates + start_fraction * distances
    directions = tip_fraction * distances
    base_coordinates = {
        cname : [cval] for cname, cval in zip(('x', 'y', 'z'), base_coordinates) 
    }
    directions = {
        dname : [dval] for dname, dval in zip(('u', 'v', 'w'), directions)
    }
    # defaults = {
    #     'name' : 'ConnCone'
    # }
    # kwargs = {**defaults, **kwargs}
    trace = go.Cone(
        **base_coordinates, **directions, **kwargs,
        showscale=False
    )
    return trace



class TraceBuilder:
    """
    Builder for programmatic creation of vector-stylized traces with
    multiple interfaces.
    More convenient creation of line traces from:
        - coordinate information
        - `Vector` instances
    """
    mode: str = 'lines'

    # trace names in legend
    # naming style sets information content of
    # the legend label string
    default_name: str = 'AnonymousVector'
    naming_style: Literal['terse', 'long'] = 'terse'

    # line styling defaults
    line_width: float = 3
    line_color: str = 'blue'
    
    # vector tip styling defaults
    start_fraction: float = 0.98
    tip_fraction: float = 0.1


    def get_line_settings(self) -> dict:
        """
        Retrieve settings dictionary from current state of instance and
        class attributes.
        """
        settings = {
            'mode' : self.mode,
            'line' : {'color' : self.line_color, 'width' : self.line_width},
            'name' : self.default_name
        }
        return settings
    
    def get_tip_settings(self) -> dict:
        """
        Retrieve tip settings from current state of instance and class
        attributes.
        """
        settings = {
            'start_fraction' : self.start_fraction,
            'tip_fraction' : self.tip_fraction,
            'name' : self.default_name
        }
        return settings

    
    def from_coordinates(self,
                         base_coordinates: Iterable[int],
                         terminal_coordinates: Iterable[int],
                         line_kwargs: Optional[dict] = None,
                         tip_kwargs: Optional[dict] = None) -> VectorTraces:
        """Build a vector-styled pair of traces from bare coordinate information."""
        # create the line trace
        line_kwargs = line_kwargs or {}
        line_kwargs = {**self.get_line_settings(), **line_kwargs}
        line_trace = build_line_trace(base_coordinates,
                                      terminal_coordinates,
                                      **line_kwargs)
        # create the tip (i.e. cone) trace
        tip_kwargs = tip_kwargs or {}
        tip_kwargs = {**self.get_tip_settings(), **tip_kwargs}
        tip_trace = build_cone_trace(base_coordinates,
                                     terminal_coordinates,
                                     **tip_kwargs)
        return VectorTraces(line=line_trace, tip=tip_trace)
    

    def from_vector(self, vector: Vector,
                    line_kwargs: Optional[dict] = None,
                    tip_kwargs: Optional[dict] = None) -> VectorTraces:
        """Build vector-styled trace pairs directly."""
        if self.naming_style == 'long':
            name = create_label(vector.dataset_ID,
                                vector.base_landmark_label,
                                vector.terminal_landmark_label)
        elif self.naming_style == 'terse':
            name = vector.dataset_ID
        else:
            raise ValueError(f'invalid naming style "{self.naming_style}"')
        
        line_kwargs = line_kwargs or {}
        tip_kwargs = tip_kwargs or {}
        for kwargs in (line_kwargs, tip_kwargs):
            if name not in kwargs:
                kwargs['name'] = name
        return self.from_coordinates(vector.base, vector.terminal,
                                     line_kwargs=line_kwargs,
                                     tip_kwargs=tip_kwargs)
    

    def from_vectors(self, vectors: Iterable[Vector],
                     line_kwargs: Optional[dict] = None,
                     tip_kwargs: Optional[dict] = None) -> list[go.Trace]:
        traces = []
        for vector in vectors:
            vectortraces = self.from_vector(vector, line_kwargs, tip_kwargs)
            traces.extend(vectortraces)
        return traces