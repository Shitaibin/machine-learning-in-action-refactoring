"""
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
@Refactored by: Shitaibin
"""

import operator
from math import log
import pickle


def create_dataset():
    dataset = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    feat_names = ['no surfacing', 'flippers']
    # change to discrete values
    return dataset, feat_names


def calc_entropy(dataset):
    n_entries = len(dataset)
    label_counts = {}
    # the the number of unique elements and their occurance
    for feat_vec in dataset:
        label = feat_vec[-1]
        if label not in label_counts.keys():
            label_counts[label] = 0
        label_counts[label] += 1

    shannon_entropy = 0.0
    for label in label_counts:
        prob = float(label_counts[label]) / n_entries
        shannon_entropy -= prob * log(prob, 2)  # log base 2
    return shannon_entropy


def split_dataset(dataset, axis, value):
    """
    Get subset, which dataset[axis] == value, and remove this axis.
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


def choose_best_feature_to_split(dataset):
    """
    :param dataset: 2d list.
    :return:
    """
    n_features = len(dataset[0]) - 1  # the last column is used for the labels
    base_entropy = calc_entropy(dataset)
    best_info_gain = 0.0
    best_feature = -1
    for i in range(n_features):  # iterate over all the features
        # get a set of unique values
        feat_unique_vals = set([item[i] for item in dataset])
        new_entropy = 0.0
        for value in feat_unique_vals:
            subset = split_dataset(dataset, i, value)
            prob = len(subset) / float(len(dataset))
            new_entropy += prob * calc_entropy(subset)
        # calculate the info gain; ie reduction in entropy
        info_gain = base_entropy - new_entropy

        if info_gain > best_info_gain:  # compare this to the best gain so far
            best_info_gain = info_gain  # if better than current best, set to best
            best_feature = i

    return best_feature  # returns an integer


def majority_class(class_list):
    """
    Choose the majority class label.
    P.S. If the two class have the same count. Choose the one appear in the latest.
    :rtype: string or int
    :param class_list:
    :return:
    """
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1

    sorted_class_count = sorted(class_count.iteritems(),
                                key=operator.itemgetter(1),
                                reverse=True)
    return sorted_class_count[0][0]


def create_tree(dataset, feat_names):
    """
    Create decision tree.
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
        return majority_class(class_list)

    # splitting
    return splitting_tree(dataset, feat_names)


def splitting_tree(dataset, feat_names):
    """
    Splitting decision tree.
    :param dataset:
    :param feat_names:
    :return:
    """
    best_feat = choose_best_feature_to_split(dataset)
    best_feat_name = feat_names[best_feat]
    decision_tree = {best_feat_name: {}}
    del (feat_names[best_feat])
    feat_unique_vals = set([item[best_feat] for item in dataset])
    for value in feat_unique_vals:
        # copy all of labels, so trees don't mess up existing labels
        decision_tree[best_feat_name][value] = create_tree(
                split_dataset(dataset, best_feat, value), feat_names[:])
    return decision_tree


def classify(decision_tree, feat_names, test_date):
    """
    Using decision tree do classification.
    :param decision_tree: 
    :param feat_names: of test_data.
    :param test_date:
    :return: 
    """
    current_classify_feat = decision_tree.keys()[0]
    child_decision_tree = decision_tree[current_classify_feat]
    feat_index = feat_names.index(current_classify_feat)
    feat_val = test_date[feat_index]
    classify_result = child_decision_tree[feat_val]
    if isinstance(classify_result, dict):  # not a leaf node.
        class_label = classify(classify_result, feat_names, test_date)
    else:
        class_label = classify_result
    return class_label


def store_decision_tree(decision_tree, filename):
    fw = open(filename, 'w')
    pickle.dump(decision_tree, fw)
    fw.close()


def grab_tree(filename):
    fr = open(filename)
    return pickle.load(fr)
