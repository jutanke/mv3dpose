import sys
sys.path.insert(0, '../')
from mv3dpose.tracking import tracking, Track
from os.path import isdir, join
from os import listdir
import json

dataset_dir = '/home/user/dataset'
output_dir = join(dataset_dir, 'poses3d')
dataset_json = join(dataset_dir, 'dataset.json')

vid_dir = join(dataset_dir, 'videos')
cam_dir = join(dataset_dir, 'cameras')

assert isdir(vid_dir)
assert isdir(cam_dir)

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

if 'valid_frames' in Settings:
    valid_frames = Settings['valid_frames']
else:
    # take all frames in the setup
    files = listdir(join(vid_dir, 'camera00'))
    print("#frames", len(files ))

# ~~~~~~~~~~~~~~~~~~~~~~


