import numpy as np
import json
from os.path import isdir, isfile, join


class MultiOpenPoseKeypoints:

    def __init__(self, pe_kpts):
        """
        :param pe_kpts: {list}
        """
        self.pe_kpts = pe_kpts

    def predict(self, frame):
        """
        :param frame:
        :return:
        """
        result = []
        for pe in self.pe_kpts:
            pred = pe.predict(frame)
            result.append(pred)
        return result


class AndreasKeypoints:

    def __init__(self, naming, loc, scale_pose=1.0):
        """
        """
        assert isdir(loc), loc
        self.scale_pose = scale_pose
        self.loc = loc
        self.naming = naming
    
    def predict(self, frame):
        fname = join(self.loc, self.naming + '.poses') % (frame, )
        assert isfile(fname), fname
        with open(fname, 'r') as f:
            kp = json.load(f)

        J = 18
        ours_vs_andreas = np.array([
            (0, 0), (1, 0), (2, 6), (3, 8), (4, 10),
            (5, 5), (6, 7), (7, 9),
            (8, 12), (9, 14), (10, 16),
            (11, 11), (12, 13), (13, 15),
            (14, 2), (15, 1), (16, 4), (17, 3)
        ])

        results = []
        for person in kp:
            OUR = ours_vs_andreas[:, 0]
            OP = ours_vs_andreas[:, 1]
            our_person = np.empty((J, 3), np.float32)
            kps = np.reshape(person, (-1, 3))
            our_person[OUR] = kps[OP]

            for i, (x, y, v) in enumerate(our_person):
                if v < 0.001:
                    our_person[i, 0] = -1
                    our_person[i, 1] = -1
                    our_person[i, 2] = -1

            results.append(our_person)

        return results

class OpenPoseKeypoints:

    def __init__(self, naming, loc):
        """
        :param naming:
        :param loc:
        """
        assert isdir(loc), loc
        self.loc = loc
        self.naming = naming

    def predict(self, frame):
        """
        :param frame:
        :return:
        """
        fname = join(self.loc, self.naming + '_keypoints.json') % (frame, )
        assert isfile(fname), fname
        with open(fname, 'r') as f:
            kp = json.load(f)

        ours_vs_openpose = np.array([
            (0, 0), (1, 1), (2, 2), (3, 3), (4, 4),
            (5, 5), (6, 6), (7, 7),
            (8, 9), (9, 10), (10, 11),
            (11, 12), (12, 13), (13, 14),
            (14, 15), (15, 16), (16, 17), (17, 18),
            (18, 19), (19, 20), (20, 21),
            (21, 22), (22, 23), (23, 24)
        ])

        # J = 18
        J = 24

        results = []
        for person in kp['people']:
            OUR = ours_vs_openpose[:, 0]
            OP = ours_vs_openpose[:, 1]
            our_person = np.empty((J, 3), np.float32)
            kps = np.reshape(person['pose_keypoints_2d'], (-1, 3))
            our_person[OUR] = kps[OP]

            our_peron[:, 0:2] *= self.scale_pose

            for i, (x, y, v) in enumerate(our_person):
                if v < 0.001:
                    our_person[i, 0] = -1
                    our_person[i, 1] = -1
                    our_person[i, 2] = -1

            results.append(our_person)

        return results
