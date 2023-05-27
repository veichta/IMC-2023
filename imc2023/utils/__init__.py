import numpy as np



def rotmat2qvec(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion. TODO: check if this is correct.

    Args:
        R (np.ndarray): Rotation matrix.

    Returns:
        np.ndarray: Quaternion vector.
    """
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def rot_mat_z(angle):
    """Takes angle in degrees and computes the rotation matrix around z axis"""
    c = np.cos(np.pi*angle/180)
    s = np.sin(np.pi*angle/180)
    return np.array([[c,s,0 ],[-s,c,0],[0,0,1]])
