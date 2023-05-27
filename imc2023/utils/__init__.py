import numpy as np
def rot_mat_z(angle):
    """Takes angle in degrees and computes the rotation matrix around z axis"""
    c = np.cos(np.pi*angle/180)
    s = np.sin(np.pi*angle/180)
    return np.array([[c,s,0 ],[-s,c,0],[0,0,1]])
