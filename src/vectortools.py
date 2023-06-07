"""
Additional tooling and utility.
"""
import numpy as np

from typing import Iterable, Union
from pathlib import Path

PathLike = Union[str, Path]


def extract_IJK_position(landmarks: Iterable[dict], label: str) -> np.ndarray:
    """
    Extract the IJK position of the landmark exactly matching the given `label`.
    If multiple landmarks with the same label are present the first encountered
    is used.
    Intended use is to extract the position information of e.g. the RoundWindow
    landmark from the collection of landmark dictionaries for a single
    dataset instance.

    Parameters
    ----------

    landmarks : iterable of dict
        Landmark information as an iterable of dictionaries.
        Must at least contain 'label' and 'ijk_position' keys.

    label : str
        Label string for the desired landmark, e.g. 'RoundWindow'

    Returns
    -------

    ijk_position : np.ndarray
        IJK position as 1D integer array of size 3.
    """
    ijk_position = None
    for landmark in landmarks:
        if landmark['label'] == label:
            ijk_position = landmark['ijk_position']
            break
    if ijk_position is None:
        message = (f'landmark with label "{label}" is not contained in given landmarks iterable')
        raise KeyError(message)
    if not isinstance(ijk_position, np.ndarray):
        ijk_position = np.array(ijk_position)
    return ijk_position


def compute_vector_coordinates(base_label: str, terminal_label: str,
                               landmarks: Iterable[dict]) -> tuple[np.ndarray]:
    """
    Compute the base and terminal coordinates for the direction vector pointing
    from the landmark indicated by the `base_label` towards the landmark
    indicated by the `terminal_label`.

    Returns
    -------

    (base, terminal) : tuple of np.ndarray
        2-tuple of base and terminal coordinates (both 1D, size 3 ndarray)
    """
    base = extract_IJK_position(landmarks, base_label)
    terminal = extract_IJK_position(landmarks, terminal_label)
    return (base, terminal)

        
def extract_shape(landmarks: Iterable[dict]) -> np.ndarray:
    shapes = set(tuple(l['shape']) for l in landmarks)
    if len(shapes) != 1:
        raise ValueError(f'Expected single shape but got N={len(shapes)}')
    shape = np.array(shapes.pop())
    return shape
