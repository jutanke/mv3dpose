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
            (14, 15), (15, 16), (16, 17), (17, 18)
        ])

        results = []
        for person in kp['people']:
            OUR = ours_vs_openpose[:, 0]
            OP = ours_vs_openpose[:, 1]
            our_person = np.empty((18, 3), np.float32)
            kps = np.reshape(person['pose_keypoints_2d'], (-1, 3))
            our_person[OUR] = kps[OP]

            for i, (x, y, v) in enumerate(our_person):
                if v < 0.001:
                    our_person[i, 0] = -1
                    our_person[i, 1] = -1
                    our_person[i, 2] = -1

            results.append(our_person)

        return results
