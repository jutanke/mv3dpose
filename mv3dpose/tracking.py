import numpy as np
from mv3dpose.baseline import estimate, distance_between_poses
from scipy.optimize import linear_sum_assignment
from scipy.ndimage.filters import gaussian_filter1d
import json


def tracking(calib_per_frame, poses_per_frame,
             actual_frames=None,
             epi_threshold=40,
             merge_distance=-1,
             last_seen_delay=2,
             distance_threshold=-1,
             correct_limb_size=True,
             scale_to_mm=1,
             min_track_length=4,
             max_distance_between_tracks=100,
             z_axis=2):
    """
    :param calib_per_frame: [ [cam1, ... ], ... ] * frames
    :param poses_per_frame: [ [pose1, ...], ... ] * frames
    :param actual_frames: [ frame1, ... ] nd.array {int}
    :param epi_threshold:
    :param scale_to_mm: d * scale_to_mm = d_in_mm
        that means: if our scale is in [m] we need to set
        scale_to_mm = 1000
    :param min_track_length:
    :param correct_limb_size: if True remove limbs that are too long or short
    :param last_seen_delay: how long can a track be "forgotten" to
        be recovered
    :param max_distance_between_tracks: maximal distance between
        two tracks in [mm]
    :param z_axis: some datasets are rotated around one axis
    :return:
    """
    # check if we only have one set of cameras
    # (cameras do not change over time)
    fixed_cameras = True
    if isinstance(calib_per_frame[0], (list, )):
        fixed_cameras = False
    n_frames = len(poses_per_frame)
    if not fixed_cameras:
        assert n_frames == len(calib_per_frame)
    if actual_frames is not None:
        assert len(actual_frames) == n_frames

    all_tracks = []

    for t in range(n_frames):
        if actual_frames is not None:
            real_t = actual_frames[t]
        else:
            real_t = t

        if fixed_cameras:
            calib = calib_per_frame
        else:
            calib = calib_per_frame[t]

        poses = poses_per_frame[t]
        assert len(poses) == len(calib)

        predictions = estimate(calib, poses,
                               scale_to_mm=scale_to_mm,
                               epi_threshold=epi_threshold,
                               merge_distance=merge_distance,
                               distance_threshold=distance_threshold,
                               correct_limb_size=correct_limb_size,
                               get_hypothesis=False)

        possible_tracks = []
        for track in all_tracks:
            if track.last_seen() + last_seen_delay < real_t:
                continue  # track is too old..
            possible_tracks.append(track)

        n = len(possible_tracks)
        if n > 0:
            m = len(predictions)
            D = np.empty((n, m))
            for tid, track in enumerate(possible_tracks):
                for pid, pose in enumerate(predictions):
                    D[tid, pid] = track.distance_to_last(pose)

            rows, cols = linear_sum_assignment(D)
            D = D * scale_to_mm  # ensure that distances in D are in [mm]

            handled_pids = set()
            for tid, pid in zip(rows, cols):
                d = D[tid, pid]
                if d > max_distance_between_tracks:
                    continue

                # merge pose into track
                track = possible_tracks[tid]
                pose = predictions[pid]
                track.add_pose(real_t, pose)
                handled_pids.add(pid)

            # add all remaining poses as tracks
            for pid, pose in enumerate(predictions):
                if pid in handled_pids:
                    continue
                track = Track(real_t, pose,
                              last_seen_delay=last_seen_delay, z_axis=z_axis)
                all_tracks.append(track)

        else:  # no tracks yet... add them
            for pose in predictions:
                track = Track(real_t, pose,
                              last_seen_delay=last_seen_delay, z_axis=z_axis)
                all_tracks.append(track)

    surviving_tracks = []
    for track in all_tracks:
        if len(track) >= min_track_length:
            surviving_tracks.append(track)

    return surviving_tracks


class Track:

    @staticmethod
    def smoothing(track, sigma,
                  interpolation_range=4,
                  relevant_jids=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]):
        """ smoothing of a track
        :param track:
        :param sigma:
        :param interpolation_range:
        :param relevant_jids: is set up for mscoco
        :return:
        """
        first_frame = track.first_frame()
        last_frame = track.last_seen() + 1
        n_frames = last_frame - first_frame
        print("n frames", n_frames)

        relevant_jids_lookup = {}
        relevant_jids = set(relevant_jids)

        delete_jids = []

        # step 0: make sure all relevent jids have entries
        for jid in relevant_jids:
            jid_found = False
            for frame in range(first_frame, last_frame):
                pose = track.get_by_frame(frame)
                if pose is not None and pose[jid] is not None:
                    jid_found = True
                    break

            if not jid_found:
                delete_jids.append(jid)

        for jid in delete_jids:
            relevant_jids.remove(jid)

        # step 1:
        unrecoverable = set()
        for jid in relevant_jids:
            XYZ = np.empty((n_frames, 3))
            for frame in range(first_frame, last_frame):
                pose = track.get_by_frame(frame)

                if pose is None or pose[jid] is None:
                    start_frame = max(first_frame, frame - interpolation_range)
                    end_frame = min(last_frame, frame + interpolation_range)

                    from_left = []
                    for _frame in range(start_frame, frame):
                        _pose = track.get_by_frame(_frame)
                        if _pose is None or _pose[jid] is None:
                            continue
                        from_left.append(_pose[jid])

                    from_right = []
                    for _frame in range(frame, end_frame):
                        _pose = track.get_by_frame(_frame)
                        if _pose is None or _pose[jid] is None:
                            continue
                        from_right.append(_pose[jid])

                    pts = []
                    if len(from_left) > 0:
                        pts.append(from_left[-1])
                    if len(from_right) > 0:
                        pts.append(from_right[0])

                    if len(pts) > 0:
                        pt = np.mean(pts, axis=0)
                    else:
                        # print("JID", jid)
                        # print('n frames', n_frames)
                        # print('current frame', frame)
                        # assert len(pts) > 0, 'jid=' + str(jid)
                        unrecoverable.add((jid, frame))
                        pt = np.array([0., 0., 0.])

                else:
                    pt = pose[jid]
                XYZ[frame - first_frame] = pt

            XYZ_sm = np.empty_like(XYZ)
            for dim in [0, 1, 2]:
                D = XYZ[:, dim]
                D = gaussian_filter1d(D, sigma, mode='reflect')
                XYZ_sm[:, dim] = D
            relevant_jids_lookup[jid] = XYZ_sm

        new_track = None

        for frame in range(first_frame, last_frame):
            person = []
            for jid in range(track.J):
                if jid in relevant_jids_lookup:
                    if (jid, frame) in unrecoverable:
                        person.append(None)
                    else:
                        XYZ_sm = relevant_jids_lookup[jid]
                        pt = XYZ_sm[frame - first_frame]
                        person.append(pt)
                else:
                    pose = track.get_by_frame(frame)
                    if pose is None:
                        person.append(None)
                    else:
                        person.append(pose[jid])
            if new_track is None:
                new_track = Track(frame, person, track.last_seen_delay, track.z_axis)
            else:
                new_track.add_pose(frame, person)

        return new_track

    def __init__(self, t, pose, last_seen_delay, z_axis):
        """
        :param t: {int} time
        :param pose: 3d * J
        :param last_seen_delay: max delay between times
        :param z_axis: some datasets are rotated around one axis
        """
        self.frames = [int(t)]
        self.J = len(pose)
        self.poses = [pose]
        self.last_seen_delay = last_seen_delay
        self.lookup = None
        self.z_axis = z_axis

    def __len__(self):
        if len(self.frames) == 1:
            return 1
        else:
            first = self.frames[0]
            last = self.frames[-1]
            return last - first + 1

    def to_file(self, fname):

        poses = []
        for p in self.poses:
            if isinstance(p, np.ndarray):
                poses.append(p.tolist())
            elif isinstance(p, list):
                pose = []
                for joint in p:
                    if isinstance(joint, np.ndarray):
                        joint = joint.tolist()
                    pose.append(joint)
                poses.append(pose)
            else:
                raise Exception('Type is not a list:' + str(type(p)))

        data = {
            "J": self.J,
            "frames": self.frames,
            "poses": poses,
            'z_axis': self.z_axis
        }
        with open(fname, 'w') as f:
            json.dump(data, f)

    def last_seen(self):
        return self.frames[-1]

    def first_frame(self):
        return self.frames[0]

    def add_pose(self, t, pose):
        """ add pose
        :param t:
        :param pose:
        :return:
        """
        last_t = self.last_seen()
        assert last_t < t
        diff = t - last_t
        assert diff <= self.last_seen_delay
        self.frames.append(t)
        self.poses.append(pose)
        self.lookup = None  # reset lookup

    def get_by_frame(self, t):
        """ :returns pose by frame
        :param t:
        :return:
        """
        if self.lookup is None:
            self.lookup = {}
            for f, pose in zip(self.frames, self.poses):
                self.lookup[f] = pose

        if t in self.lookup:
            return self.lookup[t]
        else:
            return None

    def distance_to_last(self, pose):
        """ calculates the distance to the
            last pose
        :param pose:
        :return:
        """
        last_pose = self.poses[-1]
        return distance_between_poses(pose, last_pose, self.z_axis)
