import numpy as np
from typing import List, Tuple, Dict, Optional

XYZ = ['x', 'y', 'z']


def vec2rotmag(vect: Dict[str, float]) -> Tuple[Dict[str, float], float]:
    """
    Takes a vector and breaks it down into constituent
    Euler rotations and magnitude
    """
    assert all([k in vect.keys() for k in XYZ]),\
        "Ill formed vector provided: " + str(vect)
    x = vect['x']
    y = vect['y']
    z = vect['z']
    mag = np.sqrt(x*x + y*y + z*z)
    rot = dict([[k, vect[k]/mag] for k in XYZ])
    return (rot, mag)


def rotmag2vec(rotation: Dict[str, float],
               magnitude: float) -> Dict[str, float]:
    """
    Takes a rotation [x,y,z] angle dict and magnitude,
    and returns the equivalent [x,y,z] vector
    """
    assert all([k in rotation.keys() for k in XYZ]),\
        "Ill formed vector provided: " + str(rotation)
    return dict([[k, rotation[k]*magnitude] for k in XYZ])


def rad2deg(rad: float) -> float:
    return rad * 180 / np.pi

def deg2rad(deg: float) -> float:
    return deg / 180 * np.pi
