from pak.datasets.Shelf import Shelf
from mv3dpose.geometry.camera import AffineCamera


def get(data_root, frame):
    """
        gets the data for the given frame
    :param data_root:
    :param frame:
    :return:
    """
    shelf = Shelf(data_root, verbose=False)
    X, Y, _Calib = shelf.get_frame(frame)
    Calib = []
    for P in _Calib:
        w = 1032
        h = 776
        Calib.append(AffineCamera(P, w, h))

    return X, Y, Calib

