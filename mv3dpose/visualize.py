import sys
sys.path.insert(0, '../')
import matplotlib.pyplot as plt
import cv2
import json
import numpy as np
from mv3dpose.tracking import Track
from os.path import isdir, join, isfile
from os import listdir, makedirs
import mv3dpose.geometry.camera as camera
import shutil
from tqdm import tqdm
import math


dataset_dir = '/home/user/dataset'
dataset_json = join(dataset_dir, 'dataset.json')

vid_dir = join(dataset_dir, 'videos')
cam_dir = join(dataset_dir, 'cameras')
trk_dir = join(dataset_dir, 'tracks3d')
assert isdir(trk_dir), "the tracks must be extracted!"
assert isdir(cam_dir), "could not find cameras!"
assert isdir(vid_dir), "could not find videos!"

# ~~~~~ LOAD SETTINGS ~~~~~

Settings = json.load(open(dataset_json))

n_cameras = Settings['n_cameras']
valid_frames = Settings['valid_frames']

print('CAMERAS', n_cameras)
print("#frames", len(valid_frames))

tracks = [json.load(open(join(trk_dir, f))) for f in listdir(trk_dir)]
print("#tracks", len(tracks))


# -- create lookups --
tracks_by_frame = {}
pose_by_track_and_frame = {}
for frame in valid_frames:
    assert frame not in tracks_by_frame
    tracks_by_frame[frame] = []
    for tid, track in enumerate(tracks):
        frames = track['frames']
        poses = track['poses']
        for i, t in enumerate(frames):
            if t > frame:
                break
            elif t == frame:
                tracks_by_frame[frame].append(tid)
                pose_by_track_and_frame[tid, frame] = poses[i]


def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)


# colors = get_cmap(len(tracks))
n_tracks = len(tracks)
colors = np.random.random(size=(n_tracks, 1, 3))

# ~~~~~~~~~~~~~
# C A M E R A S
# ~~~~~~~~~~~~~
print('\n[load cameras]')
Calib = []  # { n_frames x n_cameras }
for t in tqdm(valid_frames):
    calib = []
    Calib.append(calib)

    for cid in range(n_cameras):
        local_camdir = join(cam_dir, 'camera%02d' % cid)
        assert isdir(local_camdir)
        cam_fname = join(local_camdir, 'frame%09d.json' % t)
        assert isfile(cam_fname), cam_fname
        cam = camera.Camera.load_from_file(cam_fname)
        calib.append(cam)

# ====================================
# ~~~~ PLOT FRAMES ~~~~
# ====================================
output_dir = join(dataset_dir, 'visualization')
if isdir(output_dir):
    shutil.rmtree(output_dir)


LIMBS = [
    (0, 1), (0, 15), (0, 14), (15, 17), (14, 16),
    (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (2, 8), (5, 11), (8, 11),
    (8, 9), (9, 10), (10, 21), (21, 22), (22, 23),
    (11, 12), (12, 13), (13, 18), (18, 19), (19, 20)
]


makedirs(output_dir)


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


for i, frame in tqdm(enumerate(valid_frames)):

    fig = plt.figure(figsize=(16, 12))
    H = 2 if n_cameras < 8 else 3
    W = int(math.ceil(n_cameras / H))
    fname = join(output_dir, 'frame%09d.png' % i)

    tracks_on_frame = tracks_by_frame[frame]

    for cid in range(n_cameras):

        camera_img_dir = join(vid_dir, 'camera%02d' % cid)
        img_file = join(camera_img_dir, 'frame%09d.png' % frame)
        assert isfile(img_file)
        im = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)
        h, w, _ = im.shape

        cam = Calib[i][cid]
        ax = fig.add_subplot(H, W, cid+1)
        ax.axis('off')
        ax.set_xlim([0, w])
        ax.set_ylim([h, 0])
        ax.imshow(im)

        for tid in tracks_on_frame:
            color = colors[tid]
            pose3d = pose_by_track_and_frame[tid, frame]

            # we need to mask over None
            assert len(pose3d) == 24
            mask = [True] * 24
            for jid in range(24):
                if pose3d[jid] is None:
                    pose3d[jid] = [0, 0, 0]
                    mask[jid] = False
                else:
                    mm = np.mean(pose3d[jid])
                    if isclose(0., mm):
                        mask[jid] = False

            pose3d = np.array(pose3d, dtype=np.float32)

            pose2d = cam.projectPoints(pose3d)
            for jid in range(24):
                if mask[jid]:
                    x, y = pose2d[jid]
                    ax.scatter(x, y, c=color)

            for a, b in LIMBS:
                if mask[a] and mask[b]:
                    x1, y1 = pose2d[a]
                    x2, y2 = pose2d[b]
                    ax.plot([x1, x2], [y1, y2], c=np.squeeze(color))

    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
