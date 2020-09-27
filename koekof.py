from scipy.spatial.distance import cdist
import numpy as np

from commons.fkm import FuzzyKMeans


def main():
    # Q1
    data = np.array([
        [4.8, 2.3, 3.6, 3.2],
        [24.4, 20.4, 21.7, 22.1],
        [20.6, 20, 22.2, 24.1],
        [4.9, 2.7, 1.6, 4.1],
        [1.8, 2.7, 1.8, 4.6]
    ])
    c = np.array(
        [
            [4.5, 2.4, 3.3, 2.4],
            [23.2, 24.1, 24.1, 23]
        ]
    )
    d = FuzzyKMeans.center_dist(data, c)
    mu = FuzzyKMeans.membership_matrix(d)
    for index, u in enumerate(mu):
        print(f"X{index+1}: C{(np.argmax(u)+1)}")

    # Q2
    data = np.array([
        [9.3, 9.4],
        [9.6, 9.5],
        [3.8, 7],
        [9.9, 7.3]
    ])
    distance = cdist(data, data)
    distance[distance == 0] = np.inf
    print(distance.min())


if __name__ == '__main__':
    main()
