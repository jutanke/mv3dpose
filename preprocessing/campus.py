from pak.datasets.EPFL_Campus import EPFL_Campus
from mv3dpose.geometry.camera import AffineCamera


def get(data_root, frame):
    """
        gets the data for the given frame
    :param data_root:
    :param frame:
    :return:
    """
    campus = EPFL_Campus(data_root)
    X, Y, _Calib = campus.get_frame(frame)
    Calib = []
    for P in _Calib:
        w = 360
        h = 288
        Calib.append(AffineCamera(P, w, h))

    return X, Y, Calib
