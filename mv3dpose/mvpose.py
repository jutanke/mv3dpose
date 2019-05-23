import sys
sys.path.insert(0, '../')
from mv3dpose.data.openpose import OpenPoseKeypoints, MultiOpenPoseKeypoints
from mv3dpose.tracking import tracking, Track
import mv3dpose.geometry.camera as camera
from os.path import isdir, join, isfile
from os import makedirs
from tqdm import tqdm
from os import listdir
from time import time
import json
import shutil

dataset_dir = '/home/user/dataset'
output_dir = join(dataset_dir, 'tracks3d')
dataset_json = join(dataset_dir, 'dataset.json')

vid_dir = join(dataset_dir, 'videos')
cam_dir = join(dataset_dir, 'cameras')
kyp_dir = join(dataset_dir, 'poses')

assert isdir(vid_dir)
assert isdir(cam_dir)

if isdir(output_dir):
    print('\n[deleting existing output directory...]')
    shutil.rmtree(output_dir)

makedirs(output_dir)

# ~~~~~~~~~~~~~~~~~~~~~~

Settings = json.load(open(dataset_json))

scale_to_mm = Settings['scale_to_mm']
n_cameras = Settings['n_cameras']


def get_optional(key, default_value):
    if key in Settings:
        return Settings[key]
    else:
        return default_value


epi_threshold = get_optional(
    'epi_threshold', 80)
max_distance_between_tracks = get_optional(
    'max_distance_between_tracks', 80)
min_track_length = get_optional(
    'min_track_length', 10)
merge_distance = get_optional(
    'merge_distance', 80)
last_seen_delay = get_optional(
    'last_seen_delay', 15)
smoothing_sigma = get_optional(
    'smoothing_sigma', 2)
smoothing_interpolation_range = get_optional(
    'smoothing_interpolation_range', 50)
do_smoothing = get_optional(
    'do_smoothing', True
)

if 'valid_frames' in Settings:
    valid_frames = Settings['valid_frames']
else:
    # take all frames in the setup
    files = listdir(join(vid_dir, 'camera00'))
    valid_frames = list(range(len(files)))

print("\n#frames", len(valid_frames))

# ~~~~~~~~~~~~~~~~~
# K E Y P O I N T S
# ~~~~~~~~~~~~~~~~~
print('\n[load keypoints]')
keypoints = []
for cid in tqdm(range(n_cameras)):
    loc = join(kyp_dir, 'camera%02d' % cid)
    assert isdir(loc)
    pe = OpenPoseKeypoints('frame%09d', loc)
    keypoints.append(pe)
pe = MultiOpenPoseKeypoints(keypoints)

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

# ~~~~~~~~~~~~~~~~~~~~~~~
# L O A D  2 D  P O S E S
# ~~~~~~~~~~~~~~~~~~~~~~~
print('\n[load 2d poses]')
poses_per_frame = []
for frame in tqdm(valid_frames):
    predictions = pe.predict(frame)
    poses_per_frame.append(predictions)

# ~~~~~~~~~~~~~~~
# T R A C K I N G
# ~~~~~~~~~~~~~~~
print('\n[tracking]')
_start = time()
tracks = tracking(Calib, poses_per_frame,
                  epi_threshold=epi_threshold,
                  scale_to_mm=scale_to_mm,
                  max_distance_between_tracks=max_distance_between_tracks,
                  actual_frames=valid_frames,
                  min_track_length=min_track_length,
                  merge_distance=merge_distance,
                  last_seen_delay=last_seen_delay)
_end = time()
print('\telapsed', _end - _start)
print("\n\t# detected tracks:", len(tracks))

# ~~~~~~~~~~~~~~~
# S M O T H I N G
# ~~~~~~~~~~~~~~~
if do_smoothing:
    print('\n[smoothing]')
    tracks_ = []
    for track in tqdm(tracks):
        track = Track.smoothing(track,
                                sigma=smoothing_sigma,
                                interpolation_range=smoothing_interpolation_range)
        tracks_.append(track)
    tracks = tracks_

# ~~~~~~~~~~~~~~~~~
# S E R I A L I Z E
# ~~~~~~~~~~~~~~~~~
print('\n[serialize 3d tracks]')
for tid, track in tqdm(enumerate(tracks)):
    fname = join(output_dir, 'track' + str(tid) + '.json')
    track.to_file(fname)
