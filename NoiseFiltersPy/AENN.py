"""
@ref: [An Experiment with the Edited Nearest-Neighbor Rule](https://ieeexplore.ieee.org/ielx5/21/4309513/04309523.pdf)
"""

from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import typing as t

from NoiseFiltersPy.Filter import *


class AENN:
    def __init__(self, max_neighbours: int = 5, n_jobs: int = -1):
        self.max_neighbours = max_neighbours
        self.filter = Filter(parameters={"max_neighbours": self.max_neighbours})
        self.n_jobs = n_jobs

    def __call__(self, data: t.Sequence, classes: t.Sequence) -> Filter:
        self.isNoise = np.array([False] * len(classes))
        for n_neigh in range(1, self.max_neighbours + 1):
            self.clf = KNeighborsClassifier(n_neighbors=n_neigh, algorithm='kd_tree', n_jobs=self.n_jobs)
            for indx in np.argwhere(np.invert(self.isNoise)):
                self.clf.fit(np.delete(data, indx, axis=0), np.delete(classes, indx, axis=0))
                pred = self.clf.predict(data[indx])
                self.isNoise[indx] = pred != classes[indx]
            print(f"n_neigh: {n_neigh}, is_noise count:, {sum(self.isNoise)}, total: {len(self.isNoise)}")
        self.filter.rem_indx = np.argwhere(self.isNoise)
        notNoise = np.invert(self.isNoise)
        self.filter.set_cleanData(data[notNoise], classes[notNoise])
        return self.filter


if __name__ == "__main__":
    from sklearn.datasets import load_iris

    data = load_iris()
    rm_feature_id = data.feature_names.index("petal length (cm)")
    features = data.data[:, [idx for idx in range(len(data.feature_names)) if idx != rm_feature_id]]
    labels = data.target
    filter = AENN()(features, labels)
    print(filter.rem_indx.shape, filter.rem_indx[:, 0])
