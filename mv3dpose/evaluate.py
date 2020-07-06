import sys
sys.path.insert(0, '../')
from os.path import join, isdir, isfile
from os import listdir
import json
from collections import namedtuple
import numpy.linalg as la
import numpy as np
from tqdm import tqdm
from mv3dpose.tracking import Track

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
    alpha = 0.3
    L_Arms = []
    U_Arms = []
    L_Legs = []
    U_Legs = []
    GTIDs = []

    if isinstance(Y, np.ndarray):
        if len(Y.shape) == 4:  # 1, N_PPL, J, 3
            # hack to work for UMPM
            assert Y.shape[0] == 1
            Y = Y[0]

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


# =====================================
# PROGRAM START
# =====================================

dataset_dir = '/home/user/dataset'
gt_dir = join(dataset_dir, 'gt')
track_dir = join(dataset_dir, 'tracks3d')
dataset_json = join(dataset_dir, 'dataset.json')

assert isfile(dataset_json)
assert isdir(gt_dir), "GT dir:" + gt_dir
assert isdir(track_dir), "Track dir:" + track_dir


# ~~~~~ LOAD SETTINGS ~~~~~

Settings = json.load(open(dataset_json))

n_cameras = Settings['n_cameras']
valid_frames = Settings['valid_frames']
tracks = [Track.from_file(join(track_dir, f)) for f in listdir(track_dir)]

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print("#cameras", n_cameras)
print("#valid frames", len(valid_frames))
print("#tracks", len(tracks))
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('\n')

print('\nload ground truth...')
Pos3d = {}
for frame in tqdm(valid_frames):
    fname = join(gt_dir, 'frame%09d.npy' % frame)
    if isfile(fname):
        Pos3d[frame] = np.load(fname, allow_pickle=True)


print("\nfind best 3d matches...")
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


print("\nResults:")
print("=====================================")

total_avg = []
for key, values in PER_GTID.items():
    print('actor ', key)
    print('\tuarms:', np.mean(values['uarms']))
    print('\tlarms:', np.mean(values['larms']))
    print('\tulegs:', np.mean(values['ulegs']))
    print('\tllegs:', np.mean(values['llegs']))
    avg = np.mean([
        np.mean(values['uarms']),
        np.mean(values['larms']),
        np.mean(values['ulegs']),
        np.mean(values['llegs'])
    ])
    total_avg.append(avg)
    print('\tavg:  ', avg)
print('\navg*:  ', np.mean(total_avg))

print("=====================================")
