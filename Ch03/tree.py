"""
Created on Jan 7, 2016
Decision Tree Source Code
@author: Shitaibin
"""
from collections import Counter
from math import log

import numpy as np


class DecisionTree:
    """A decision tree classifier.
    """

    def __init__(self, criterion="id3"):
        self.n_features_ = None
        self.tree = {}
        self.criterion = criterion
        self.__criterion_method_dict__ = {"id3": self.__choose_by_id3__,
                                          "c4.5": self.__choose_by_c45__}

    def fit(self, X, y, feat_names):
        """Build a decision tree from the training set(X, y)

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values (class labels in classification).

        Returns
        -------
        self : object
            Returns self.
        """
        # combine X,y to get dataset
        # way 1, abandon: array to list, will make anything as string
        # y_list = [[yy] for yy in y]
        # y_array = np.asarray(y_list)
        # X_array = np.asarray(X)
        # dataset_array = np.hstack((X_array, y_array))
        # dataset = dataset_array.tolist()

        # X or y is array
        if isinstance(X, np.ndarray):
            X = X.tolist()
        if isinstance(y, np.ndarray):
            y = y.tolist()

        # way 2, adopt:iteration
        dataset = []
        for x, yy in zip(X, y):
            # way 1: easy to read
            # TODO: bug, should not change x
            # x.append(yy)
            # dataset.append(x[:])

            # way 2: not easy to read
            dataset.append(x + [yy])

        # call create_tree and save result to self.tree
        self.tree = self.__create_tree__(dataset, feat_names)

    def predict(self, X, feat_names):
        """Predict class or regression value for X.
        For a classification model, the predicted class for each sample in X is
        returned.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples.

        feat_names : list.

        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted classes.
        """
        # X is a list of x, return a list of class labels
        if isinstance(X[0], list):
            return [self.__classify__(self.tree, feat_names, x) for x in X]
        else:
            # X is x, return the class label
            return self.__classify__(self.tree, feat_names, X)

    def __calc_entropy__(self, dataset):
        n_entries = len(dataset)
        # the the number of unique elements and their occurrence
        # label_counts = {}
        # for feat_vec in dataset:
        #     label = feat_vec[-1]
        #     if label not in label_counts.keys():
        #         label_counts[label] = 0
        #     label_counts[label] += 1

        labels = [item[-1] for item in dataset]
        label_counts = Counter(labels)

        shannon_entropy = 0.0
        for label in label_counts:
            prob = float(label_counts[label]) / n_entries
            shannon_entropy -= prob * log(prob, 2)  # log base 2
        return shannon_entropy

    def __get_dataset__(self, dataset, axis, value):
        """ Get subset, which dataset[axis] == value, and remove this axis.

        :param dataset: 2d list.
        :param axis: int, feature id.
        :param value: feature value.
        :return: 2d list.
        """
        subset = []
        for feature_vector in dataset:
            if feature_vector[axis] == value:
                reduced_feature_vector = feature_vector[:axis]  # chop out axis used for splitting
                reduced_feature_vector.extend(feature_vector[axis + 1:])
                subset.append(reduced_feature_vector)
        return subset

    def __choose_best_feature_to_split__(self, dataset):
        """

        :param dataset: 2d list.
        :return:
        """
        # way1: normal
        # if self.criterion == "id3":
        #     return self.__choose_by_id3__(dataset)
        # else:
        #     return self.__choose_by_c45__(dataset)

        # way2: Pythonic
        return self.__criterion_method_dict__.get(self.criterion)(dataset)

    def __choose_by_c45__(self, dataset):
        """Choose best feature by information gain ratio.

        :param dataset:
        :return:
        """
        n_features = len(dataset[0]) - 1  # the last column is used for the labels
        base_entropy = self.__calc_entropy__(dataset)
        best_info_gain_ratio = 0.0
        best_feature = -1
        for i in range(n_features):  # iterate over all the features
            # get a set of unique values
            feat_unique_vals = set([item[i] for item in dataset])
            new_entropy = 0.0
            split_info = 0.0
            for value in feat_unique_vals:
                subset = self.__get_dataset__(dataset, i, value)
                prob = len(subset) / float(len(dataset))
                new_entropy += prob * self.__calc_entropy__(subset)
                split_info -= prob * log(prob, 2)
            # calculate the info gain; ie reduction in entropy
            info_gain = base_entropy - new_entropy
            info_gain_ratio = info_gain / split_info

            if info_gain_ratio > best_info_gain_ratio:  # compare this to the best gain so far
                best_info_gain_ratio = info_gain_ratio  # if better than current best, set to best
                best_feature = i
        return best_feature  # returns an integer

    def __choose_by_id3__(self, dataset):
        """Choose best feature by information gain.

        :param dataset:
        :return:
        """
        n_features = len(dataset[0]) - 1  # the last column is used for the labels
        base_entropy = self.__calc_entropy__(dataset)
        best_info_gain = 0.0
        best_feature = -1
        for i in range(n_features):  # iterate over all the features
            # get a set of unique values
            feat_unique_vals = set([item[i] for item in dataset])
            new_entropy = 0.0
            for value in feat_unique_vals:
                subset = self.__get_dataset__(dataset, i, value)
                prob = len(subset) / float(len(dataset))
                new_entropy += prob * self.__calc_entropy__(subset)
            # calculate the info gain; ie reduction in entropy
            info_gain = base_entropy - new_entropy

            if info_gain > best_info_gain:  # compare this to the best gain so far
                best_info_gain = info_gain  # if better than current best, set to best
                best_feature = i
        return best_feature  # returns an integer

    def __majority_class__(self, class_list):
        """Choose the majority class label.

        P.S. If the two class have the same count. Choose any one is okay.
        :rtype: string or int
        :param class_list:
        :return:
        """
        # v2
        # class_count = {}
        # for vote in class_list:
        #     if vote not in class_count.keys():
        #         class_count[vote] = 0
        #     class_count[vote] += 1
        #
        # sorted_class_count = sorted(class_count.iteritems(),
        #                             key=operator.itemgetter(1),
        #                             reverse=True)
        # return sorted_class_count[0][0]

        # v3
        return Counter(class_list).keys()[0]

    def __create_tree__(self, dataset, feat_names):
        """Create decision tree.

        :param dataset: 2d list.
        :param feat_names: targets
        :return: class label or dict of child tree.
        """
        class_list = [item[-1] for item in dataset]

        # stop splitting when all of the classes are equal
        if len(set(class_list)) == 1:
            return class_list[0]

        # stop splitting when there are no more features in dataset
        if len(dataset[0]) == 1:
            return self.__majority_class__(class_list)

        # splitting
        return self.__splitting_tree__(dataset, feat_names)

    def __splitting_tree__(self, dataset, feature_names):
        """Splitting decision tree.

        :param dataset:
        :param feature_names:
        :return:
        """
        best_feature = self.__choose_best_feature_to_split__(dataset)
        best_feat_name = feature_names[best_feature]
        decision_tree = {best_feat_name: {}}
        del (feature_names[best_feature])
        feat_unique_vals = set([item[best_feature] for item in dataset])
        for value in feat_unique_vals:
            # copy all of labels, so trees don't mess up existing labels
            decision_tree[best_feat_name][value] = self.__create_tree__(
                    self.__get_dataset__(dataset, best_feature, value), feature_names[:])
        return decision_tree

    def __classify__(self, decision_tree, feature_names, test_date):
        """Using decision tree do classification.

        :param decision_tree:
        :param feature_names: of test_data.
        :param test_date:
        :return:
        """

        current_feature = decision_tree.keys()[0]
        child_decision_tree = decision_tree[current_feature]
        feature_index = feature_names.index(current_feature)
        feature_val = test_date[feature_index]
        classify_result = child_decision_tree[feature_val]
        if isinstance(classify_result, dict):  # classify in child tree
            class_label = self.__classify__(classify_result, feature_names, test_date)
        else:
            class_label = classify_result
        return class_label
