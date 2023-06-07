"""
Additional tooling and utility to load and wrangle-transform
landmark data from HDF5 files on storage towards vector representation.
"""
import warnings
import numpy as np

from copy import deepcopy
from typing import Iterable, Union
from pathlib import Path

from nettools.pathutils import parse_filepath, FilenameParsingError
from nettools.loader import peek_dataset

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



def create_augmented_landmarks(peek_info: dict) -> list[dict]:
    """
    Validate and transform the peeking information from a single dataset
    instance to iterable of landmark dictionaries.

    Checks for matching shapes and creates a list of dictionaries
    with singular shape entry.
    """
    # TODO: hardcoding names is bad
    rawshape, _ = peek_info['raw-0']
    labelshape, _ = peek_info['label-0']
    if rawshape != labelshape:
        raise ValueError(f'shape mismatch: got {rawshape} for raw shape and '
                         f'{labelshape} for label shape')
    augmented_landmarks = []
    for landmark in peek_info['landmarks']:
        augmented_landmark = deepcopy(landmark)
        augmented_landmark['shape'] = np.array(rawshape)
        augmented_landmarks.append(augmented_landmark)
    return augmented_landmarks


def collect_dataset_landmarks(directory: PathLike) -> dict[str, list[dict]]:
    """
    Crawl a directory for dataset instances as HDF5 files and create a mapping
    from basestem -> list of augmented landmark dictionaries.

    The intention is to use the resulting dictionary for programmatic instantiation
    of direction vectors between landmarks for many instances of the dataset.
    """
    directory = Path(directory)
    dataset_landmarks = {}
    for item in directory.iterdir():
        try:
            sempath = parse_filepath(item)
        except FilenameParsingError:
            warnings.warn(
                f'Could not parse item: "{item}" :: item skipped'
            )
        
        dataset_landmarks[sempath.basestem] = create_augmented_landmarks(
            peek_dataset(sempath)    
        )
    return dataset_landmarks
        
        
        
def extract_shape(landmarks: Iterable[dict]) -> np.ndarray:
    shapes = set(tuple(l['shape']) for l in landmarks)
    if len(shapes) != 1:
        raise ValueError(f'Expected single shape but got N={len(shapes)}')
    shape = np.array(shapes.pop())
    return shape




