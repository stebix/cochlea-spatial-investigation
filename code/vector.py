"""
Package enables capturing and displaying information about
landmark orientation in IJK voxel space.
"""
import dataclasses
import numpy as np

from typing import Optional, Literal

from vectortools import compute_vector_coordinates, extract_shape


@dataclasses.dataclass
class Vector:
    """Vector between two landmarks of a specific dataset."""
    # semantic information
    dataset_ID: str
    base_landmark_label: str
    terminal_landmark_label: str
    # numerical coordinate information
    base: np.ndarray        # vector base coordinates
    terminal: np.ndarray    # vector terminal coordinates
    shape: np.ndarray       # shape of "host" voxel volume
    # publicly settable only for compatibility
    delta: Optional[np.ndarray] = None
    
    _rotation_state: int = 0
    _Rz90 = np.array(
        [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    )
        
    def __post_init__(self) -> None:
        self.delta = self.terminal - self.base
        if self._rotation_state % 4 == 0:
            self._rotation_state = 0
    
    @property
    def norm(self) -> float:
        return np.linalg.norm(self.delta)
    
    
    def zrotate90(self) -> 'Vector':
        """
        Rotate coordinate information around `z-axis` by 90° or pi/2.
        Note that coordinate system (i.e. raw voxel volume is rotated alongside!)
        """
        shape = np.array([self.shape[1], self.shape[0], self.shape[2]])
        
        base_prime = self._Rz90 @ self.base
        base = base_prime + np.where(base_prime < 0, shape, 0)
        
        terminal_prime = self._Rz90 @ self.terminal
        terminal = terminal_prime + np.where(terminal_prime < 0, shape, 0)
        
        rotation_state = self._rotation_state + 1
        
        return Vector(dataset_ID=self.dataset_ID,
                      base_landmark_label=self.base_landmark_label,
                      terminal_landmark_label=self.terminal_landmark_label,
                      shape=shape, base=base, terminal=terminal,
                      _rotation_state=rotation_state)

    @classmethod
    def unit_vector(cls, axis: Literal['x', 'y', 'z']) -> 'Vector':
        """Construct unit vector along the provided axis."""
        axis = axis.capitalize()
        coordinates = {
            'X' : [1, 0, 0], 'Y' : [0, 1, 0], 'Z' : [0, 0, 1]
        }
        ID = f'{axis}AXIS'
        base_label = 'ORIGIN'
        terminal_label = f'{axis}DIR'
        base = np.zeros(3)
        terminal = np.array(coordinates[axis])
        shape = np.ones(3, dtype=np.int32)
        unit_vector = cls(dataset_ID=ID, base_landmark_label=base_label,
                          terminal_landmark_label=terminal_label,
                          base=base, terminal=terminal, shape=shape)
        return unit_vector

    @classmethod
    def unit_X(cls) -> 'Vector':
        return cls.unit_vector('x')
        
    @classmethod
    def unit_Y(cls) -> 'Vector':
        return cls.unit_vector('y')

    @classmethod
    def unit_Z(cls) -> 'Vector':
        return cls.unit_vector('z')




def as_normalized_origin_vector(vector: Vector) -> Vector:
    """Re-create vector as normalized and origin-bound."""
    updates = {
        'base' : np.zeros(3),
        'terminal' : vector.delta / vector.norm
    }
    # transfer previous data but update base and terminal coordinates
    #
    previous = dataclasses.asdict(vector)
    kwargs = {**previous, **updates}
    vector = Vector(**kwargs)
    return vector


def inner(v1: Vector, v2: Vector, /) -> float:
    """Compute inner product."""
    return np.dot(v1.delta, v2.delta)
    

def angle(v1: Vector, v2: Vector, /, *, retunit='deg') -> float:
    """Compute the angle between two instances of `Vector`."""
    if retunit == 'deg':
        convert = np.rad2deg
    elif retunit == 'rad':
        convert = lambda x: x
    else:
        raise ValueError(f'invalid retunit argument: "{retunit}"')
    return convert(np.arccos(inner(v1, v2) / (v1.norm * v2.norm)))


def create_Rz(alpha: float) -> np.ndarray:
    """Create rotation matrix `R_z` around z-axis by angle 
       alpha [degree]."""
    alpha = np.deg2rad(alpha)
    cosalpha = np.cos(alpha)
    sinalpha = np.sin(alpha)
    mat = [[cosalpha, -sinalpha, 0],
           [sinalpha, cosalpha, 0],
           [0, 0, 1]]
    return np.array(mat)


def rotate_rodrigues(v: np.ndarray, e: np.ndarray, theta: float) -> np.ndarray:
    """
    Rotate vector `v` around axis `e` (i.e. arbitrary unit vector ||e|| = 1)
    by the angle `theta` [degree] according to Olinde Rodrigues' formula.
    """
    sintheta = np.sin(np.deg2rad(theta))
    costheta = np.cos(np.deg2rad(theta))
    # compute summands separately
    s1 = v * costheta
    s2 = np.cross(e, v) * sintheta
    s3 = e * (np.dot(v, e)) * (1 - costheta)
    return s1 + s2 + s3



def build_vectors(dataset_landmarks: dict[str, list[dict]],
                  base_landmark_label: str,
                  terminal_landmark_label: str) -> list[Vector]:
    """
    Build a list of connection vectors pointing from the base landmark
    to the terminal landmark for all instances provided in the `dataset_landmarks`.

    This mapping is assumed to be of the form:
     dataset_ID (str) -> [landmark-1 (dict), landmark-2 (dict), ...] 
    """
    vectors = []
    for dataset_ID, landmarks in dataset_landmarks.items():
        base, terminal = compute_vector_coordinates(base_landmark_label,
                                                    terminal_landmark_label,
                                                    landmarks)
        shape = extract_shape(landmarks)
        vectors.append(
            Vector(dataset_ID=dataset_ID, base_landmark_label=base_landmark_label,
                   terminal_landmark_label=terminal_landmark_label, base=base,
                   terminal=terminal, shape=shape)
        )
    return vectors