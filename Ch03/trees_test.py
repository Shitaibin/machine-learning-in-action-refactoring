import trees
import unittest
from math import log

class TreesTestCase(unittest.TestCase):
    """
    Unittest for trees.py.
    """
    def test_calc_entropy(self):
        data, labels = trees.create_dataset()
        entropy = -(0.4 * log(0.4, 2) + 0.6 * log(0.6, 2))
        self.assertEqual(entropy, trees.calc_entropy(data))

    def test_split_dateset(self):
        """

        :return:
        """
        # test 1
        data = [[1, 2],
                [1, 3],
                [2, 3]]
        axis = 0
        val = 1
        res = [[2], [3]]
        self.assertEqual(res, trees.split_dataset(data, axis, val))

        # test 2
        axis = 1
        val = 2
        res = [[1]]
        self.assertEqual(res, trees.split_dataset(data, axis, val))

    def test_choose_best_feature_to_split(self):
        """
        :return:
        """
        # In this test. test data without label column.

        # positive test
        # test 1, best feature is feature 0
        data = [[1, 2],
                [1, 3],
                [2, 3]]
        best_featrue = 0
        self.assertEqual(best_featrue, trees.choose_best_feature_to_split(data))

        # negtive test
        # test 2, data is empty
        data = []
        self.assertRaises(IndexError, lambda: trees.choose_best_feature_to_split(data))

        def test_majority_cnt(self):
            """

            :param self:
            :return:
            """

if __name__ == "__main__":
    unittest.main()