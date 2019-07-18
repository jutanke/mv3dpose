import numpy as np
import cv2
from os import makedirs
from os.path import join, isdir, isfile
from pak.util.download import download
from pak.util.unzip import unzip
from math import ceil, floor
from mv3dpose.geometry.camera import AffineCamera


def get(data_root, frame):
    """
    :param data_root:
    :param frame: starting at frame 0
    :return:
    """
    seq_zipname = 'player2sequence1.zip'
    seq_dir = 'Sequence 1'
    player = 2

    root = join(data_root, 'football2')
    root = join(root, 'player' + str(player))
    if not isdir(root):
        makedirs(root)

    seq_url = 'http://www.csc.kth.se/cvap/cvg/MultiViewFootballData/' + seq_zipname
    seq_dir = join(root, seq_dir)

    if not isdir(seq_dir):
        seq_zip = join(root, seq_zipname)
        if not isfile(seq_zip):
            print('downloading... ', seq_url)
            download(seq_url, seq_zip)

        print('unzipping... ', seq_zip)
        unzip(seq_zip, root)

    pos2d_file = join(seq_dir, 'positions2d.txt')
    pos2d = np.loadtxt(pos2d_file)
    N = 14  # number joints
    C = 3  # number cameras
    T = len(pos2d) / 2 / N / C
    assert floor(T) == ceil(T)
    T = int(T)

    pos2d_result = np.zeros((2, N, C, T))
    counter = 0
    for t in range(T):
        for c in range(C):
            for n in range(N):
                for i in range(2):
                    pos2d_result[i, n, c, t] = pos2d[counter]
                    counter += 1
    pos2d = pos2d_result

    # ~~~ pos3d ~~~
    pos3d_file = join(seq_dir, 'positions3d.txt')
    assert isfile(pos3d_file)
    pos3d = np.loadtxt(pos3d_file)
    pos3d_result = np.zeros((3, N, T))
    assert T == int(len(pos3d) / 3 / N)
    counter = 0
    for t in range(T):
        for n in range(N):
            for i in range(3):
                pos3d_result[i, n, t] = pos3d[counter]
                counter += 1
    pos3d = pos3d_result

    # ~~~ Cameras ~~~
    cam_file = join(seq_dir, 'cameras.txt')
    assert isfile(cam_file)
    cams = np.loadtxt(cam_file)
    cameras = np.zeros((2, 4, C, T))
    assert T == int(len(cams) / 2 / 4 / C)

    counter = 0
    for t in range(T):
        for c in range(C):
            for j in range(4):
                for i in range(2):
                    cameras[i, j, c, t] = cams[counter]
                    counter += 1

    Im = []
    h = -1; w = -1
    for cam in ['Camera 1', 'Camera 2', 'Camera 3']:
        im_dir = join(seq_dir, cam)
        assert isdir(im_dir)
        im_name = join(im_dir, "%05d.png" % (frame+1))
        assert isfile(im_name)
        im = cv2.cvtColor(cv2.imread(im_name), cv2.COLOR_BGR2RGB)
        Im.append(im)

        if w == -1 or h == -1:
            assert h == -1 and w == -1
            h, w, _ = im.shape
        else:
            h_, w_, _ = im.shape
            assert h_ == h and w_ == w

    Im = np.array(Im)

    Calib = []
    for cid in [0, 1, 2]:
        cam = np.zeros((3, 4))
        cam[0:2, :] = cameras[:, :, cid, frame]
        cam[2,3] = 1
        Calib.append(AffineCamera(cam, w, h))

    # h x w x cam
    Pts2d = []
    for cid in [0, 1, 2]:
        d2d = pos2d[:,:,cid, frame]
        Pts2d.append(d2d)

    d3d = pos3d[:, :, frame]

    return Im, np.transpose(d3d), Calib
