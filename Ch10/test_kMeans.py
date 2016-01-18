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


if __name__ == "__main__":
    unittest.main()
