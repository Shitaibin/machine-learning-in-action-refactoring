"""
Created on Jan 4, 2016
Decision Tree Unittest Source Code for trees.py.
@author: Shitaibin
"""

import unittest
from math import log

import trees


class TreesTestCase(unittest.TestCase):
    """
    Unittest for trees.py.
    """

    def test_calc_entropy(self):
        data, feat_names = trees.create_dataset()
        entropy = -(0.4 * log(0.4, 2) + 0.6 * log(0.6, 2))
        self.assertEqual(entropy, trees.calc_entropy(data))

    def test_split_dateset(self):
        """

        :return:
        """
        # In this test. test data without label column.
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

        # positive test
        # test 1, best feature is feature 0
        data = [[1, 2, 'yes'],
                [1, 3, 'yes'],
                [2, 3, 'no']]
        best_feature = 0
        self.assertEqual(best_feature, trees.choose_best_feature_to_split(data))

        # test 2, best feature is feature 0
        data = [[1, 2, 'yes'],
                [2, 3, 'no'],
                [3, 3, 'no']]
        best_feature = 0
        self.assertEqual(best_feature, trees.choose_best_feature_to_split(data))

        # negative test
        # test 3, data is empty
        data = []
        self.assertRaises(IndexError, lambda: trees.choose_best_feature_to_split(data))

    def test_majority_class(self):
        """
        unittest for majority_class.
        :param self:
        :return:
        """
        class_list = ['yes', 'no', 'no', 'yes']
        majority_class = 'yes'
        self.assertEqual(majority_class, trees.majority_class(class_list))

        class_list = ['yes', 'no', 'no', 'yes', 'no']
        majority_class = 'no'
        self.assertEqual(majority_class, trees.majority_class(class_list))

        class_list = ['yes', 'no', 'no', 'yes', 'no']
        majority_class = 'no'
        self.assertEqual(majority_class, trees.majority_class(class_list))

        class_list = ['yes', 'no', 'no', 'yes', 'maybe']
        majority_class = 'yes'
        self.assertEqual(majority_class, trees.majority_class(class_list))

        class_list = ['yes', 'maybe', 'no', 'yes', 'maybe']
        majority_class = 'maybe'
        self.assertEqual(majority_class, trees.majority_class(class_list))

    def test_create_tree(self):
        """
        Unittest for create tree.
        :return:
        """
        data, feat_names = trees.create_dataset()
        decision_tree = {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
        self.assertEqual(decision_tree, trees.create_tree(data, feat_names))

        data = [[1, 2, 'yes'],
                [1, 3, 'yes'],
                [2, 3, 'no']]
        feat_names = ['no surfacing', 'flippers']
        decision_tree = {'no surfacing': {1: 'yes', 2: 'no'}}
        self.assertEqual(decision_tree, trees.create_tree(data, feat_names))

    def test_classify(self):
        """
        Unittest for function classify.
        :return: classification result.
        """
        # test 1: training data
        item = [1, 0]
        feat_names = ['no surfacing', 'flippers']
        result = 'no'
        decision_tree = {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
        self.assertEqual(result, trees.classify(decision_tree, feat_names, item))

        # test 2: training data with different feat_names
        item = [0, 1]
        feat_names = ['flippers', 'no surfacing']
        result = 'no'
        decision_tree = {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
        self.assertEqual(result, trees.classify(decision_tree, feat_names, item))

        # test 3: not training data
        item = [0, 0]
        feat_names = ['flippers', 'no surfacing']
        result = 'no'
        decision_tree = {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
        self.assertEqual(result, trees.classify(decision_tree, feat_names, item))
        pass


if __name__ == "__main__":
    unittest.main()
