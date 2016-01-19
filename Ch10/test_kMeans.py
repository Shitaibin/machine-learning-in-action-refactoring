import unittest

import numpy as np
from numpy import (array)
from numpy.linalg import norm

from kMeans import distance, load_dataset


class KMeansTestCase(unittest.TestCase):
    """

    """

    def test_load_dataset(self):
        dataset = [[1, 2], [3, 4]]
        file_name = "temp_test_kmeans.txt"
        f = open(file_name, "w")
        lines = ["{}\t{}\n".format(line[0], line[1]) for line in dataset]
        f.writelines(lines)
        f.close()

        dataset_loaded = load_dataset(file_name)
        self.assertIsInstance(dataset_loaded, np.matrix)
        self.assertEqual(dataset, dataset_loaded.tolist())

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
        # I find it's also hard to test. Because the initial centroids is randomly.
        pass


if __name__ == "__main__":
    unittest.main()
