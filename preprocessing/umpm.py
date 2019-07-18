from pak.datasets.UMPM import UMPM
from mv3dpose.geometry.camera import ProjectiveCamera
import json
import numpy as np
Settings = json.load(open('settings.txt'))
root = Settings['data_root']
user = Settings['UMPM']['username']
pwd = Settings['UMPM']['password']

umpm = UMPM(root, user, pwd)

dataset = 'p2_chair_2'
_X_, _Y_, _Calib_ = umpm.get_data('p2_free_1')


def transform2kth(Y):
    """ For PCP calculation we utilize the KTH joints
    # R_ANKLE       0
    # R_KNEE        1
    # R_HIP         2
    # L_HIP         3
    # L_KNEE        4
    # L_ANKLE       5
    # R_WRIST       6
    # R_ELBOW       7
    # R_SHOULDER    8
    # L_SHOULDER    9
    # L_ELBOW       10
    # L_WRIST       11
    # BOTTOM_HEAD   12
    # TOP_HEAD      13
    :param Y: [ n_frames x 30 x 5 ]
    :return:
    """
    # each Y contains two people a 15 joints
    J, dim = Y.shape
    assert J == 30, str(J)
    assert dim == 5, str(dim)

    result = np.empty((2, 14, 3), np.float32)

    a2kth = np.array([13, 11, 9, 10, 12, 14,
                      5, 4, 3, 6, 7, 8,
                      1, 2])  # first person -> kth
    b2kth = np.array([24, 26, 28, 25, 27, 29,
                      20, 19, 18, 21, 22, 23,
                      16, 17])  # second person -> kth

    result[0, :] = Y[a2kth, :3]
    result[1, :] = Y[b2kth, :3]

    return result


# X_r = X['r']  # l r s f


def get(_, frame):
    """
    :param _:
    :param frame:
    :return:
    """
    w = 644
    h = 486
    Im = []
    Y = []
    Calib = []

    for cid in ['l', 'r', 's', 'f']:
        im = _X_[cid][frame]
        Im.append(im)

        cam = _Calib_[cid]
        cam = ProjectiveCamera(np.array(cam['K']),
                               np.array(cam['rvec']),
                               np.array(cam['tvec']),
                               np.array(cam['distCoeff']),
                               w, h)
        Calib.append(cam)

    Y.append(transform2kth(_Y_[frame]))
    Im = np.array(Im)
    return Im, Y, Calib
