import numpy as np
import numpy.linalg as la
from mv3dpose.hypothesis import Hypothesis, HypothesisList, get_believe
from scipy.optimize import linear_sum_assignment
from mv3dpose.data.default_limbs import DEFAULT_SENSIBLE_LIMB_LENGTH


def distance_between_poses(pose1, pose2, z_axis):
    """
    :param pose1:
    :param pose2:
    :param z_axis: some datasets are rotated around one axis
    :return:
    """
    J = len(pose1)
    assert len(pose2) == J
    distances = []
    for jid in range(J):
        if pose2[jid] is None or pose1[jid] is None:
            continue
        d = la.norm(pose2[jid] - pose1[jid])
        distances.append(d)

    if len(distances) == 0:
        # TODO check this heuristic
        # take the centre distance in x-y coordinates
        valid1 = []
        valid2 = []
        for jid in range(J):
            if pose1[jid] is not None:
                valid1.append(pose1[jid])
            if pose2[jid] is not None:
                valid2.append(pose2[jid])

        assert len(valid1) > 0
        assert len(valid2) > 0
        mean1 = np.mean(valid1, axis=0)
        mean2 = np.mean(valid2, axis=0)
        assert len(mean1) == 3
        assert len(mean2) == 3

        # we only care about xy coordinates
        mean1[z_axis] = 0
        mean2[z_axis] = 0

        return la.norm(mean1 - mean2)
    else:
        return np.mean(distances)  # TODO try different versions


def estimate(calib, poses,
             scale_to_mm=1,
             merge_distance=-1,
             epi_threshold=40,
             distance_threshold=-1,
             correct_limb_size=True,
             z_axis=2,
             get_hypothesis=False):
    """
    :param calib:
    :param poses:
    :param scale_to_mm: d * scale_to_mm = d_in_mm
        that means: if our scale is in [m] we need to set
        scale_to_mm = 1000
    :param merge_distance: in [mm]
    :param epi_threshold:
    :param z_axis: some datasets are rotated around one axis
    :param correct_limb_size: if True remove limbs that are too long or short
    :param get_hypothesis:
    :return:
    """
    n_cameras = len(calib)
    assert n_cameras == len(poses)
    #assert n_cameras >= 3, 'Expect multi-camera setup'

    # ~~~~~~~~~~~~~~~~~~~~~
    # cleanup
    poses_ = []
    for cid in range(len(calib)):
        cam_ = []
        loc_pred = poses[cid]
        poses_.append(cam_)
        for pose in loc_pred:
            if get_believe(pose) > 0.30:
                cam_.append(pose)
    poses = poses_
    # ~~~~~~~~~~~~~~~~~~~~~

    # add all detections in the first frames as hypothesis
    # TODO: handle the case when there is NO pose in 1. cam
    first_cid = 0
    H = [
        Hypothesis(pose, calib[0], epi_threshold,
                   scale_to_mm=scale_to_mm,
                   distance_threshold=distance_threshold,
                   debug_2d_id=(first_cid, pid))
        for pid, pose in enumerate(poses[first_cid])]

    for cid in range(1, n_cameras):
        cam = calib[cid]
        all_detections = poses[cid]

        n_hyp = len(H)
        n_det = len(all_detections)

        C = np.zeros((n_hyp, n_det))
        Mask = np.zeros_like(C).astype('int32')

        for pid, person in enumerate(all_detections):
            for hid, h in enumerate(H):
                cost, veto = h.calculate_cost(person, cam)
                C[hid, pid] = cost
                if veto:
                    Mask[hid, pid] = 1

        rows, cols = linear_sum_assignment(C)

        handled_pids = set()
        for hid, pid in zip(rows, cols):
            is_masked = Mask[hid, pid] == 1
            handled_pids.add(pid)
            if is_masked:
                # even the closest other person is
                # too far away (> threshold)
                H.append(Hypothesis(
                    all_detections[pid],
                    cam,
                    epi_threshold,
                    scale_to_mm=scale_to_mm,
                    distance_threshold=distance_threshold,
                    debug_2d_id=(cid, pid)))
            else:
                H[hid].merge(all_detections[pid], cam)
                H[hid].debug_2d_ids.append((cid, pid))

        for pid, person in enumerate(all_detections):
            if pid not in handled_pids:
                H.append(Hypothesis(
                    all_detections[pid],
                    cam,
                    epi_threshold,
                    scale_to_mm=scale_to_mm,
                    distance_threshold=distance_threshold,
                    debug_2d_id=(cid, pid)))

    surviving_H = []
    humans = []
    for hyp in H:
        if hyp.size() > 1:
            humans.append(hyp.get_3d_person())
            surviving_H.append(hyp)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # merge closeby poses
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if merge_distance > 0:
        distances = []  # (hid1, hid2, distance)
        n = len(humans)
        for i in range(n):
            for j in range(i+1, n):
                pose1 = humans[i]
                pose2 = humans[j]
                distance = distance_between_poses(pose1, pose2, z_axis)
                distances.append((i, j, distance * scale_to_mm))

        # the root merge is always the smallest hid
        # go through all merges and point higher hids
        # towards their smallest merge hid

        mergers_root = {}  # hid -> root
        mergers = {}  # root: [ hid, hid, .. ]
        all_merged_hids = set()
        for hid1, hid2, distance in distances:
            if distance > merge_distance:
                continue

            if hid1 in mergers_root and hid2 in mergers_root:
                continue  # both are already handled

            if hid1 in mergers_root:
                hid1 = mergers_root[hid1]

            if hid1 not in mergers:
                mergers[hid1] = [hid1]

            mergers[hid1].append(hid2)
            mergers_root[hid2] = hid1
            all_merged_hids.add(hid1)
            all_merged_hids.add(hid2)

        merged_surviving_H = []
        merged_humans = []

        for hid in range(n):
            if hid in mergers:
                hyp_list = [surviving_H[hid2] for hid2 in mergers[hid]]
                hyp = HypothesisList(hyp_list)
                pose = hyp.get_3d_person()
                merged_surviving_H.append(hyp)
                merged_humans.append(pose)
            elif hid not in all_merged_hids:
                merged_surviving_H.append(surviving_H[hid])
                merged_humans.append(humans[hid])

        humans = merged_humans
        surviving_H = merged_surviving_H
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if correct_limb_size:

        # --- remove limbs with bad length ---
        ua_range = DEFAULT_SENSIBLE_LIMB_LENGTH[2]
        la_range = DEFAULT_SENSIBLE_LIMB_LENGTH[3]
        ul_range = DEFAULT_SENSIBLE_LIMB_LENGTH[7]
        ll_range = DEFAULT_SENSIBLE_LIMB_LENGTH[8]

        for human in humans:
            # check left arm
            if test_distance(human, scale_to_mm, 5, 6, *ua_range):
                human[6] = None
                human[7] = None  # we need to disable hand too
            elif test_distance(human, scale_to_mm, 6, 7, *la_range):
                human[7] = None

            # check right arm
            if test_distance(human, scale_to_mm, 2, 3, *ua_range):
                human[3] = None
                human[4] = None  # we need to disable hand too
            elif test_distance(human, scale_to_mm, 3, 4, *la_range):
                human[4] = None

            # check left leg
            if test_distance(human, scale_to_mm, 11, 12, *ul_range):
                human[12] = None
                human[13] = None  # we need to disable foot too
            elif test_distance(human, scale_to_mm, 12, 13, *ll_range):
                human[13] = None

            # check right leg
            if test_distance(human, scale_to_mm, 8, 9, *ul_range):
                human[9] = None
                human[10] = None  # we need to disable foot too
            elif test_distance(human, scale_to_mm, 9, 10, *ll_range):
                human[10] = None
    # ------------------------------------

    if get_hypothesis:
        return humans, surviving_H
    else:
        return humans


def test_distance(human, scale_to_mm, jid1, jid2, lower, higher):
    """
    :param human: [ (x, y, z) ] * J
    :param scale_to_mm:
    :param jid1:
    :param jid2:
    :param lower:
    :param higher:
    :return:
    """
    a = human[jid1]
    b = human[jid2]
    if a is None or b is None:
        return False
    distance = la.norm(a - b) * scale_to_mm
    if lower <= distance <= higher:
        return False
    else:
        return True
