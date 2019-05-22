import sys
sys.path.insert(0, '../')
import preprocessing.shelf as shelf
import preprocessing.kth2_football as kth
from os.path import isdir, join
from os import makedirs
import shutil
import cv2
import numpy as np
from tqdm import tqdm
import json

root = "/media/tanke/Data2/datasets"
assert isdir(root)

# ~~~~~ SHELF ~~~~~
get = shelf.get
valid_frames = list(range(300, 600))
output_dir = './../output/shelf'
data_root = join(root, 'pak')
scale_to_mm = 1000

# ~~~~~ KTH2 Football ~~~~~
get = kth.get
valid_frames = list(range(0, 214))
output_dir = './../output/kth2'
data_root = root
scale_to_mm = 1000
# ~~~~~~~~~~~~~~~~~


if isdir(output_dir):
    shutil.rmtree(output_dir)

makedirs(output_dir)

cam_dir = join(output_dir, 'cameras')
makedirs(cam_dir)
vid_dir = join(output_dir, 'videos')
makedirs(vid_dir)
gt_dir = join(output_dir, 'gt')
makedirs(gt_dir)
dataset_fname = join(output_dir, 'dataset.json')

n_cameras = -1
for t in tqdm(valid_frames):
    X, Y, Calib = get(data_root, t)
    if n_cameras == -1:
        n_cameras = len(Calib)
        for cid in range(n_cameras):
            folder_name = 'camera%02d' % cid
            makedirs(join(vid_dir, folder_name))
            makedirs(join(cam_dir, folder_name))
    else:
        assert n_cameras == len(Calib)

    for cid in range(n_cameras):
        folder_name = 'camera%02d' % cid
        im = X[cid]
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        fname = join(join(vid_dir, folder_name), 'frame%09d.png' % t)
        cv2.imwrite(fname, im)

        fname = join(join(cam_dir, folder_name), 'frame%09d.json' % t)
        cam = Calib[cid]
        cam.to_file(fname)

        fname = join(gt_dir, 'frame%09d.npy' % t)
        np.save(fname, Y)

dataset_data = {
    'n_cameras': n_cameras,
    'scale_to_mm': scale_to_mm
}


with open(dataset_fname, 'w') as f:
    json.dump(dataset_data, f)
