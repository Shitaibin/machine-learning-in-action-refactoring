'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
'''
from math import log
import operator


def create_dataset():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    # change to discrete values
    return dataSet, labels


def calc_entropy(dataset):
    n_entries = len(dataset)
    label_counts = dict()
    # the the number of unique elements and their occurance
    for feat_vec in dataset:
        label = feat_vec[-1]
        if label not in label_counts.keys():
            label_counts[label] = 0
        label_counts[label] += 1

    shannon_entropy = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / n_entries
        shannon_entropy -= prob * log(prob, 2)  # log base 2
    return shannon_entropy


def split_dataset(dataset, axis, value):
    """
    Get subset, which dataset[axis] == value, and remove this axis.
    :param dataset: 2d array.
    :param axis: int, feature id.
    :param value: feature value.
    :return: 2d array.
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
    :param dataset: 2d array.
    :return:
    """
    n_features = len(dataset[0]) - 1  # the last column is used for the labels
    base_entropy = calc_entropy(dataset)
    best_info_gain = 0.0
    best_feature = -1
    for i in range(n_features):  # iterate over all the features
        # create a list of all the examples of this feature
        feat_list = [example[i] for example in dataset]
        unique_vals_of_feature = set(feat_list)  # get a set of unique values
        new_entropy = 0.0
        for value in unique_vals_of_feature:
            subset = split_dataset(dataset, i, value)
            prob = len(subset) / float(len(dataset))
            new_entropy += prob * calc_entropy(subset)
        # calculate the info gain; ie reduction in entropy
        infoGain = base_entropy - new_entropy
        if infoGain > best_info_gain:  # compare this to the best gain so far
            best_info_gain = infoGain  # if better than current best, set to best
            best_feature = i
    return best_feature  # returns an integer


def majority_cnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(
        classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def create_tree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]  # stop splitting when all of the classes are equal
    # stop splitting when there are no more features in dataSet
    if len(dataSet[0]) == 1:
        return majority_cnt(classList)
    bestFeat = choose_best_feature_to_split(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # copy all of labels, so trees don't mess up existing labels
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = create_tree(
            split_dataset(dataSet, bestFeat, value), subLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel


def store_tree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


def grab_tree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
