import sys
sys.path.insert(0, '../')
from mv3dpose.data.openpose import OpenPoseKeypoints, MultiOpenPoseKeypoints
from mv3dpose.tracking import tracking, Track
import mv3dpose.geometry.camera as camera
from os.path import isdir, join, isfile
from os import listdir
from tqdm import tqdm
from collections import namedtuple
import json
from itertools import permutations
import numpy as np
import numpy.linalg as la

# =====================================
# FUNCTIONS
# =====================================


def evaluate(gt, d, alpha):
    """
        percentage of correctly estimated parts.
        This score only works on single-human estimations
        and the 3d data must be transformed to fit the
        KTH football2 data format (see {transform3d_from_mscoco})
    :param gt: ground truth human
    :param d: detection human
    :param alpha: 0.5
    :return:
    """
    assert len(gt) == 14
    assert len(d) == 14
    result = namedtuple('Result', [
        'upper_arms',
        'lower_arms',
        'lower_legs',
        'upper_legs',
        'all_parts'])

    limbs = [(7, 6), (10, 11)]  # -- lower arms --
    result.lower_arms = calculate_pcp_for_limbs(alpha, d, gt, limbs)

    limbs = [(8, 7), (9, 10)]  # -- upper arms --
    result.upper_arms = calculate_pcp_for_limbs(alpha, d, gt, limbs)

    limbs = [(0, 1), (5, 4)]  # -- lower legs --
    result.lower_legs = calculate_pcp_for_limbs(alpha, d, gt, limbs)

    limbs = [(1, 2), (3, 4)]  # -- upper legs --
    result.upper_legs = calculate_pcp_for_limbs(alpha, d, gt, limbs)

    result.all_parts = (result.lower_arms + result.upper_arms + result.lower_legs + result.upper_legs) / 4

    return result


def calculate_pcp_for_limbs(alpha, d, gt, limbs):
    """
        helper function
    :param alpha:
    :param d:
    :param gt:
    :param limbs:
    :return:
    """
    val = 0
    for a, b in limbs:
        s_hat = gt[a]
        s = d[a]
        e_hat = gt[b]
        e = d[b]
        if s is not None and e is not None:
            s_term = la.norm(s_hat - s)
            e_term = la.norm(e_hat - e)
            term = (s_term + e_term) / 2
            alpha_term = alpha * la.norm(s_hat - e_hat)
            if term <= alpha_term:
                val += 1/len(limbs)
    return val


def transform3d_from_mscoco(humans):
    """
        transforms the humans in the list from the mscoco
        data structure to the kth football2 structure
    :param humans: [ [ (x,y,z), ... ] * n_limbs ] * n_humans
    :return:
    """
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
    human_t = []

    for human in humans:
        new_human = [None] * 14
        new_human[0] = human[10]
        new_human[1] = human[9]
        new_human[2] = human[8]
        new_human[3] = human[11]
        new_human[4] = human[12]
        new_human[5] = human[13]
        new_human[6] = human[4]
        new_human[7] = human[3]
        new_human[8] = human[2]
        new_human[9] = human[5]
        new_human[10] = human[6]
        new_human[11] = human[7]
        new_human[12] = human[1]

        top_head = None
        nose = human[0]
        eyer = human[14]
        eyel = human[15]
        earr = human[16]
        earl = human[17]
        top_head_items = [elem for elem in [nose, eyel, eyer, earr, earl]
                          if elem is not None]
        if len(top_head_items) > 0:
            top_head_items = np.array(top_head_items)
            assert len(top_head_items.shape) == 2
            top_head = np.mean(top_head_items, axis=0)
        new_human[13] = top_head
        human_t.append(new_human)

    return human_t


def proper_pcp_calc(Y, Humans):
    alpha = 0.5
    L_Arms = []
    U_Arms = []
    L_Legs = []
    U_Legs = []
    GTIDs = []

    for gtid, gt in enumerate(Y):
        if gt is None:
            continue

        larms = 0
        uarms = 0
        llegs = 0
        ulegs = 0
        avg = 0
        for d in Humans:
            r = evaluate(gt, d, alpha)
            larms_ = r.lower_arms
            uarms_ = r.upper_arms
            ulegs_ = r.upper_legs
            llegs_ = r.lower_legs
            avg_ = (larms_ + uarms_ + ulegs_ + llegs_) / 4
            if avg_ > avg:
                avg = avg_
                larms = larms_
                uarms = uarms_
                llegs = llegs_
                ulegs = ulegs_

        L_Arms.append(larms)
        U_Arms.append(uarms)
        L_Legs.append(llegs)
        U_Legs.append(ulegs)
        GTIDs.append(gtid)

    return L_Arms, U_Arms, L_Legs, U_Legs, GTIDs


dataset_dir = '/home/user/dataset'
output_dir = join(dataset_dir, 'tracks3d')
dataset_json = join(dataset_dir, 'dataset.json')
gt_dir = join(dataset_dir, 'gt')

vid_dir = join(dataset_dir, 'videos')
cam_dir = join(dataset_dir, 'cameras')
kyp_dir = join(dataset_dir, 'poses')

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

# ~~~~~~~~~~~~~~~~~~~~~~~
# G R O U N D  T R U T H
# ~~~~~~~~~~~~~~~~~~~~~~~
print('\n[load ground truth]')
Pos3d = {}
for frame in tqdm(valid_frames):
    fname = join(gt_dir, 'frame%09d.npy' % frame)
    if isfile(fname):
        Pos3d[frame] = np.load(fname, allow_pickle=True)

cam_permutations = list(permutations(range(n_cameras)))

LA_per_perm = []
UA_per_perm = []
LL_per_perm = []
UL_per_perm = []
AVG_per_perm = []


for perm in tqdm(cam_permutations):

    Calib__ = []
    poses_per_frame__ = []
    for frame, _ in enumerate(valid_frames):
        calib = Calib[frame]  # [{Camera}] x n_cams
        predictions = poses_per_frame[frame] # [{Pred}] x n_cams
        calib__ = []
        predictions__ = []
        for cid in perm:
            calib__.append(calib[cid])
            predictions__.append(predictions[cid])
        Calib__.append(calib__)
        poses_per_frame__.append(predictions__)

    # ~~~~~~~~~~~~~~~
    # T R A C K I N G
    # ~~~~~~~~~~~~~~~
    tracks = tracking(Calib__, poses_per_frame__,
                      epi_threshold=epi_threshold,
                      scale_to_mm=scale_to_mm,
                      max_distance_between_tracks=max_distance_between_tracks,
                      actual_frames=valid_frames,
                      min_track_length=min_track_length,
                      merge_distance=merge_distance,
                      last_seen_delay=last_seen_delay)

    # ~~~~~~~~~~~~~~~
    # S M O T H I N G
    # ~~~~~~~~~~~~~~~
    if do_smoothing:
        tracks_ = []
        for track in tqdm(tracks):
            track = Track.smoothing(track,
                                    sigma=smoothing_sigma,
                                    interpolation_range=smoothing_interpolation_range)
            tracks_.append(track)
        tracks = tracks_

    PER_GTID = {}
    for frame in tqdm(valid_frames):
        Humans = []
        for track in tracks:
            pose = track.get_by_frame(frame)
            if pose is not None:
                Humans.append(pose)
        Humans = transform3d_from_mscoco(Humans)

        Y = Pos3d[frame]
        L_Arms, U_Arms, L_Legs, U_Legs, GTIDs = proper_pcp_calc(Y, Humans)
        if len(L_Arms) > 0:
            for gtid, larms, uarms, llegs, ulegs in zip(
                    GTIDs, L_Arms, U_Arms, L_Legs, U_Legs
            ):
                if not gtid in PER_GTID:
                    PER_GTID[gtid] = {
                        'larms': [],
                        'uarms': [],
                        'llegs': [],
                        'ulegs': [],
                        'frame': []
                    }
                PER_GTID[gtid]['larms'].append(larms)
                PER_GTID[gtid]['uarms'].append(uarms)
                PER_GTID[gtid]['llegs'].append(llegs)
                PER_GTID[gtid]['ulegs'].append(ulegs)
                PER_GTID[gtid]['frame'].append(frame)

    uaL = []
    laL = []
    ulL = []
    llL = []
    avgL = []
    for key, values in PER_GTID.items():
        uaL.append(np.mean(values['uarms']))
        laL.append(np.mean(values['larms']))
        ulL.append(np.mean(values['ulegs']))
        llL.append(np.mean(values['llegs']))
        avg = np.mean([
            np.mean(values['uarms']),
            np.mean(values['larms']),
            np.mean(values['ulegs']),
            np.mean(values['llegs'])
        ])
        avgL.append(avg)

    LA_per_perm.append(np.mean(uaL))
    UA_per_perm.append(np.mean(laL))
    LL_per_perm.append(np.mean(ulL))
    UL_per_perm.append(np.mean(llL))
    AVG_per_perm.append(np.mean(avgL))

print("AVG", AVG_per_perm)
np.savetxt(join(dataset_dir, 'AVG.txt'), AVG_per_perm)
np.savetxt(join(dataset_dir, 'LA.txt'), LA_per_perm)
np.savetxt(join(dataset_dir, 'UA.txt'), UA_per_perm)
np.savetxt(join(dataset_dir, 'LL.txt'), LL_per_perm)
np.savetxt(join(dataset_dir, 'UL.txt'), UL_per_perm)
