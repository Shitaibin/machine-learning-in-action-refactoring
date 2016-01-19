import unittest

import numpy as np
from numpy import (array, )
from numpy.linalg import norm

from kMeans import distance


class KMeansTestCase(unittest.TestCase):
    """

    """

    def test_distance(self):
        vector_xs = np.asarray([[1],
                                [1, 2],
                                [1, 2, 3]])
        vector_ys = np.asarray([[2],
                                [2, 3],
                                [2, 1, 4]])
        expected_results = [norm(array(x) - array(y)) for x, y in zip(vector_xs, vector_ys)]
        results = [distance(array(x), array(y)) for x, y in zip(vector_xs, vector_ys)]
        self.assertEqual(expected_results, results)

    def test_create_centroids(self):
        # hard to test, there is random in create centroids
        pass

    def test_kmeans(self):
        # TODO: unittest for kmeans
        # I find it's also hard to test. Because the initial centroids is randomly.
        pass


if __name__ == "__main__":
    unittest.main()
