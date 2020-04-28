import cv2
import numpy as np
import numpy.linalg as la
import mv3dpose.geometry.geometry as gm
from mv3dpose.geometry.stereo import get_fundamental_matrix


def get_single_human3d(humans3d):
    J = len(humans3d[0])
    human3d = [None] * J  # single 3d person
    for jid in range(J):
        pts3d = []
        for person3d in humans3d:
            if person3d[jid] is not None:
                pts3d.append(person3d[jid])

        if len(pts3d) > 0:
            pt3d = np.mean(pts3d, axis=0)
            human3d[jid] = pt3d
    return human3d


def get_distance3d(person1, person2):
    J = len(person1)
    assert len(person2) == J
    result = []
    for jid in range(J):
        if person1[jid] is None or person2[jid] is None:
            continue
        d = la.norm(person1[jid] - person2[jid])
        result.append(d)
    return np.array(result)


def merge3d(persons3d, weights):
    """
    :param person3d:
    :param weights:
    :return:
    """
    assert len(persons3d) == len(weights)
    assert 1.001 > np.sum(weights) > 0.99
    J = len(persons3d[0])
    n = len(persons3d)
    result = [None] * J
    for jid in range(J):
        w_acc = 0
        never_hit = True
        pt3d = np.array([0, 0, 0], np.float32)
        for i in range(n):
            if persons3d[i][jid] is not None:
                w = weights[i]
                w_acc += w
                pt3d += w * persons3d[i][jid]
                never_hit = False
        if not never_hit:
            pt3d = pt3d / w_acc
            result[jid] = pt3d
    return result


def calculate_cost(cam1, person1, cam2, person2):
    """ calculate the epipolar distance between two humans
    :param cam1:
    :param person1:
    :param cam2:
    :param person2:
    :return:
    """
    F = get_fundamental_matrix(cam1.P, cam2.P)
    J = len(person1)
    assert J == len(person2)

    # drop all points that are -1 -1 (not visible)
    pts1 = []
    pts2 = []
    weights1 = []
    weights2 = []
    for jid in range(J):
        x1, y1, w1 = person1[jid]
        x2, y2, w2 = person2[jid]
        if x1 >= 0 and x2 >= 0:
            pts1.append((x1, y1))
            weights1.append(w1)
            pts2.append((x2, y2))
            weights2.append(w2)
    weights1 = np.clip(weights1, a_min=0, a_max=1)
    weights2 = np.clip(weights2, a_min=0, a_max=1)

    if len(pts1) == 0:
        return np.finfo(np.float32).max

    pts1 = np.array(pts1)
    pts2 = np.array(pts2)

    epilines_1to2 = np.squeeze(
        cv2.computeCorrespondEpilines(pts1, 1, F))

    epilines_2to1 = np.squeeze(
        cv2.computeCorrespondEpilines(pts2, 2, F))

    total = 0

    n_pairs = len(pts1)
    assert n_pairs == len(pts2)

    if n_pairs == 1:
        epilines_1to2 = np.expand_dims(epilines_1to2, axis=0)
        epilines_2to1 = np.expand_dims(epilines_2to1, axis=0)

    for p1, l1to2, w1, p2, l2to1, w2 in zip(
            pts1, epilines_1to2, weights1,
            pts2, epilines_2to1, weights2):
        d1 = gm.line_to_point_distance(*l1to2, *p2)
        d2 = gm.line_to_point_distance(*l2to1, *p1)
        total += d1 + d2
    return total / n_pairs  # normalize


class Hypothesis:

    def __init__(self, pts, cam, threshold,
                 scale_to_mm,
                 distance_threshold,
                 debug_2d_id=None):
        """
        :param pts: [ (x, y, w), ... ] * J
        :param cam: ProjectiveCamera
        :param threshold: if cost is larger then this
            value then the 'other' must not be merged
        :param scale_to_mm: d * scale_to_mm = d_in_mm
            that means: if our scale is in [m] we need to set
            scale_to_mm = 1000
        :param variance_threshold: in [mm]. When exceeded we try
            to drop the joint that causes the problem
        """
        self.scale_to_mm = scale_to_mm
        self.distance_threshold = distance_threshold
        self.points = [pts]
        self.cams = [cam]
        self.threshold = threshold
        if debug_2d_id is not None:  # only for debugging
            self.debug_2d_ids = [debug_2d_id]

    def size(self):
        return len(self.points)

    def get_3d_person(self):
        assert self.size() > 1
        distance_threshold = self.distance_threshold
        scale_to_mm = self.scale_to_mm

        humans2d = []
        for cid, (cam, human) in enumerate(zip(self.cams, self.points)):
            human2d = Person2d(cid, cam, human)
            humans2d.append(human2d)

        strong_humans2d = []
        weak_humans2d = []
        for person in humans2d:
            if person.believe > 0.35:  # .45
                strong_humans2d.append(person)
            else:
                weak_humans2d.append(person)

        if len(strong_humans2d) < 2:
            strong_humans2d = humans2d
            weak_humans2d = []

        strong_humans3d = []
        n = len(strong_humans2d)
        for pid1 in range(n - 1):
            for pid2 in range(pid1 + 1, n):
                person1 = strong_humans2d[pid1]
                person2 = strong_humans2d[pid2]
                person3d, _ = person1.triangulate(person2)
                strong_humans3d.append(person3d)

        strong_human3d = get_single_human3d(strong_humans3d)

        merge_targets = []
        for weak_human2d in weak_humans2d:
            humans3d_normal = []
            for strong_human2d in strong_humans2d:
                person3d, _ = weak_human2d.triangulate(strong_human2d)
                humans3d_normal.append(person3d)
            human3d_normal = get_single_human3d(humans3d_normal)

            humans3d_mirror = []
            weak_human2d_mirror = Person2d.flip_lr(weak_human2d)
            for strong_human2d in strong_humans2d:
                person3d, _ = weak_human2d_mirror.triangulate(strong_human2d)
                humans3d_mirror.append(person3d)
            human3d_mirror = get_single_human3d(humans3d_mirror)

            d_normal = np.mean(
                get_distance3d(strong_human3d, human3d_normal) * scale_to_mm)
            d_mirror = np.mean(
                get_distance3d(strong_human3d, human3d_mirror) * scale_to_mm)

            human3d_select = human3d_normal
            if d_normal > d_mirror:
                human3d_select = human3d_mirror

            if distance_threshold > 0:
                if min(d_normal, d_mirror) < distance_threshold:
                    merge_targets.append(human3d_select)
            else:
                merge_targets.append(human3d_select)  # always choose

        n = len(merge_targets)
        if n > 0:
            weights = [1] + [1/(n+1)] * n
            weights = np.array(weights)
            weights = weights/np.sum(weights)

            human3d = merge3d([strong_human3d] + merge_targets, weights)
        else:
            human3d = strong_human3d

        return human3d

    def calculate_cost(self, o_points, o_cam):
        """
        :param o_points: other points * J
        :param o_cam: other camera
        :return:
        """
        veto = False  # if true we cannot join {other} with this
        total_cost = 0
        for person, cam in zip(self.points, self.cams):
            cost = calculate_cost(cam, person,
                                  o_cam, o_points)
            total_cost += cost
            if cost > self.threshold and get_believe(person) > 0.5:
                veto = True

        return total_cost / len(self.points), veto

    def merge(self, o_points, o_cam):
        """ integrate {other} into our hypothesis
        :param o_points:
        :param o_cam:
        :return:
        """
        self.cams.append(o_cam)
        self.points.append(o_points)


class HypothesisList:

    def __init__(self, hypothesis_list):
        """
        :param hypothesis_list: list of hypothesis'
        """
        self.hypothesis_list = hypothesis_list

    def get_3d_person(self):
        """
        :return:
        """
        poses = []
        for hyp in self.hypothesis_list:
            pose = hyp.get_3d_person()
            poses.append(pose)

        J = len(poses[0])
        result = [None] * J
        for jid in range(J):
            valid_points = []
            for pose in poses:
                if pose[jid] is not None:
                    valid_points.append(pose[jid])
            if len(valid_points) > 0:
                result[jid] = np.mean(valid_points, axis=0)
            else:
                result[jid] = None

        return result


def get_believe(points2d):
    believe = []
    J = len(points2d)
    for jid in range(J):
        w = points2d[jid, 2]
        if w >= 0:
            believe.append(w)
    return np.mean(believe)


class Person2d:

    @staticmethod
    def flip_lr(person):
        """ creates a new person with left and right flipped
        :param person: {Person2d}
        :return:
        """
        #left  = [5, 6, 7, 11, 12, 13, 15, 17, 18, 19, 20]
        #right = [2, 3, 4,  8,  9, 10, 14, 16, 21, 22, 23]
        left  = [5, 6, 7, 11, 12, 13, 15, 17]
        right = [2, 3, 4,  8,  9, 10, 14, 16]
        
        lr = left + right
        rl = right + left

        points2d = person.points2d.copy()
        points2d[lr] = points2d[rl]

        new_person = Person2d(person.cid, person.cam, points2d)
        return new_person

    def __init__(self, cid, cam, points2d, noundistort=False):
        """
        :param cid
        :param cam: {Camera}
        :param points2d: distorted points
        :param noundistort: if True do not undistort
        """
        self.cid = cid
        self.cam = cam
        self.believe = get_believe(points2d)

        if noundistort:
            self.points2d = points2d
        else:
            # ~~~ undistort ~~~
            valid_points2d = []
            jids = []
            for jid, pt2d in enumerate(points2d):
                if pt2d[0] < 0:
                    continue
                jids.append(jid)
                valid_points2d.append(pt2d)
            valid_points2d = np.array(valid_points2d, np.float32)
            points2d_undist = cam.undistort_points(valid_points2d)
            self.points2d = points2d.copy()
            for idx, jid in enumerate(jids):
                self.points2d[jid] = points2d_undist[idx]
            # ~~~~~~~~~~~~~~~~~~~~

    def __len__(self):
        return 18

    def triangulate(self, other):
        """
        :param other: {Person2d}
        :return:
        """
        Pts1 = []
        Pts2 = []
        jids = []
        W1 = []
        W2 = []

        J = len(other)
        assert J == len(self.points2d)
        assert J == len(self)

        for jid in range(J):
            if self.points2d[jid, 2] > 0 and \
                    other.points2d[jid, 2] > 0:
                Pts1.append(self.points2d[jid, 0:2])
                Pts2.append(other.points2d[jid, 0:2])
                jids.append(jid)
                W1.append(self.points2d[jid, 2])
                W2.append(other.points2d[jid, 2])

        Pts1 = np.transpose(Pts1)
        Pts2 = np.transpose(Pts2)

        Points3d = [None] * J
        w = [-1] * J
        if len(Pts1) > 0:
            Pts3d = gm.from_homogeneous(
                np.transpose(cv2.triangulatePoints(
                    self.cam.P, other.cam.P, Pts1, Pts2)))

            for jid, pt3d, w1, w2 in zip(jids, Pts3d, W1, W2):
                Points3d[jid] = pt3d
                w[jid] = min(w1, w2)

        return Points3d, np.array(w)
