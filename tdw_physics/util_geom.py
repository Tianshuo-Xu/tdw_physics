import numpy as np
import os
import copy


def save_obj(vertices: np.ndarray, faces: np.ndarray, filepath: str):
    with open(filepath, 'w') as f:
        f.write("# OBJ file\n")
        for v in vertices:
            f.write("v %.4f %.4f %.4f\n" % (v[0],v[1],v[2]))
        for face in faces:
            f.write("f")
            for vertex in face:
                f.write(" %d" % (vertex + 1))
            f.write("\n")



