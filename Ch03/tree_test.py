"""
Created on Jan 7, 2016
Decision Tree Unittest Source Code for tree.py.
@author: Shitaibin
"""

import unittest

import numpy as np

from tree import DecisionTree


class DecisionTreeTestCase(unittest.TestCase):
    """Unittest for tree.DecsionTree
    """

    def setUp(self):
        self.decision_tree = DecisionTree()

    def tearDown(self):
        self.decision_tree = None

    def test_fit(self):
        # test data
        X = [[1, 1],
             [1, 1],
             [1, 0],
             [0, 1],
             [0, 1]]
        y = ["yes", "yes", "no", "no", "no"]
        # X and y is list object
        feat_names = ['no surfacing', 'flippers']
        decision_tree = {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
        self.decision_tree.fit(X, y, feat_names)
        self.assertEqual(self.decision_tree.tree, decision_tree)

        # X and y is array
        feat_names = ['no surfacing', 'flippers']
        self.decision_tree.fit(np.asarray(X), np.asarray(y), feat_names)
        self.assertEqual(self.decision_tree.tree, decision_tree)

    def test_predict(self):
        # test 1: training data
        item = [1, 0]
        feat_names = ['no surfacing', 'flippers']
        result = 'no'
        decision_tree = {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
        self.decision_tree.tree = decision_tree
        self.assertEqual(result, self.decision_tree.predict(item, feat_names))

        # test 2: training data with different feat_names
        dataset = [[0, 1],
                   [0, 0]]
        feat_names = ['flippers', 'no surfacing']
        result = ["no", "no"]
        decision_tree = {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
        self.decision_tree.tree = decision_tree
        self.assertEqual(result, self.decision_tree.predict(dataset, feat_names))


class DecisionTreeC45TestCase(unittest.TestCase):
    """

    """

    def setUp(self):
        self.decision_tree = DecisionTree("c4.5")

    def tearDown(self):
        self.decision_tree = None

    def test_fit(self):
        # test data
        X = [[1, 1],
             [1, 1],
             [1, 0],
             [0, 1],
             [0, 1]]
        y = ["yes", "yes", "no", "no", "no"]
        # X and y is list object
        feat_names = ['no surfacing', 'flippers']
        decision_tree = {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
        self.decision_tree.fit(X, y, feat_names)
        self.assertEqual(self.decision_tree.tree, decision_tree)

        # X and y is array
        feat_names = ['no surfacing', 'flippers']
        self.decision_tree.fit(np.asarray(X), np.asarray(y), feat_names)
        self.assertEqual(self.decision_tree.tree, decision_tree)

    def test_predict(self):
        # There is no need to test predict.
        # Because, predict is not about criterion, in test_predict.
        pass


if __name__ == "__main__":
    unittest.main()
