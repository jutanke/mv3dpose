import sys
sys.path.insert(0, '../')
from os.path import isdir, isfile, join
from os import makedirs, remove
import numpy as np
import shutil
import json
import mv3dpose.geometry.camera as camera

nframes = 300
w = 1280
h = 720
cdir = '/media/tanke/Data3/datasets/action_in_kitchen/190502'
assert isdir(cdir), cdir

outdir = '/media/tanke/Data3/mv3dpose/aic_demo/cameras'

for cid in range(12):
    ex_cam = join(cdir, 'cam_ex%02d.txt' % cid)
    in_cam = join(cdir, 'cam_in%02d.txt' % cid)
    assert isfile(ex_cam) and isfile(in_cam)
    param_in = np.loadtxt(in_cam)
    K = param_in[:9].reshape(3, 3)
    dist = param_in[9:]
    # dist = np.zeros((5, 1))

    param_ex = np.loadtxt(ex_cam)
    rvec = param_ex[:3]
    tvec = param_ex[3:]

    camdir = join(outdir, 'camera%02d' % cid)
    if isdir(camdir):
        shutil.rmtree(camdir)

    makedirs(camdir)

    cam = camera.ProjectiveCamera(K, rvec, tvec, dist, w, h)

    # cam = {
    #     "K": K.tolist(),
    #     "rvec": rvec.tolist(),
    #     "tvec": rvec.tolist(),
    #     "distCoef": dist.tolist(),
    #     "w": w,
    #     "h": h
    # }

    for i in range(1, nframes + 1):
        fname = join(camdir, 'frame%09d.json' % i)
        cam.to_file(fname)
        # with open(fname, 'w') as f:
        #     json.dump(cam, f)


# dump dataset.json
fname = '/media/tanke/Data3/mv3dpose/aic_demo/dataset.json'
if isfile(fname):
    remove(fname)

dataset = {
    'n_cameras': 12,
    'scale_to_mm': 1000,
    'valid_frames': list(range(1, nframes + 1))
}

with open(fname, 'w') as f:
    json.dump(dataset, f)
