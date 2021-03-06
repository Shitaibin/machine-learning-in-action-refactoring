"""
Created on Feb 16, 2011
k Means Clustering for Ch10 of Machine Learning in Action
@author: Peter Harrington
@Refactored by: Shitaibin
"""
import json
import urllib
from time import sleep

import matplotlib.pyplot as plt
# from numpy import *
from numpy import (inf, mat, zeros, nonzero, mean, random,
                   sin, arccos, cos, pi)

from numpy.linalg import norm


def load_dataset(file_name):  # general function to parse tab -delimited floats
    """Load dataset from file_name.

    :param file_name: string.
    :return: np.matrix.
    """
    data_matrix = []  # assume last column is target value
    fr = open(file_name)
    for line in fr.readlines():
        current_line = line.strip().split('\t')
        float_line = map(float, current_line)  # map all elements to float()
        data_matrix.append(float_line)
    return mat(data_matrix)


def distance(vector_x, vector_y):
    """Euclid distance between two points.

    :param vector_x: np.ndarray
    :param vector_y: np.ndarray
    :return: float.
    """
    # return sqrt(sum(power(vector_x - vector_y, 2)))  # la.norm(vector_x-vector_y)
    return norm(vector_x - vector_y)


def create_centroids(dataset, k):
    """ Create initial centroids randomly.

    :param dataset: np.ndarray
    :param k:
    :return: np.mat
    """
    # n = shape(dataset)[1]
    dataset = mat(dataset)
    n = dataset.shape[1]
    centroids = mat(zeros((k, n)))  # create centroid mat
    for j in range(n):  # create random cluster centers, within bounds of each dimension
        min_j = min(dataset[:, j])
        range_j = float(max(dataset[:, j]) - min_j)
        centroids[:, j] = mat(min_j + range_j * random.rand(k, 1))
    return centroids


def kmeans(dataset, k, get_distance=distance, get_init_centroids=create_centroids):
    """

    :param dataset: array-like.
    :param k: int.
    :param get_distance: function.
    :param get_init_centroids:
    :return:
    """
    # m = shape(dataset)[0]
    dataset = mat(dataset)
    m = dataset.shape[0]
    cluster_assignment = mat(zeros((m, 2)))  # create mat to assign data points
    # to a centroid, also holds SE of each point
    centroids = get_init_centroids(dataset, k)
    stop = False
    while not stop:
        n_changed_points = reassign_points(centroids, cluster_assignment,
                                           dataset, get_distance, k, m)
        stop = should_stop(m, n_changed_points)
        # print centroids
        recalculate_means(centroids, cluster_assignment, dataset, k)
    return centroids, cluster_assignment


def recalculate_means(centroids, cluster_assignment, dataset, k):
    """

    :param centroids: np.matrix. old centroids
    :param cluster_assignment: np.matrix.
    :param dataset: np.matrix
    :param k: int.
    :return:
    """
    for centroid_id in range(k):  # recalculate centroids
        points_in_cluster = dataset[
            nonzero(cluster_assignment[:, 0].A == centroid_id)[0]]  # get all the point in this cluster
        centroids[centroid_id, :] = mean(points_in_cluster, axis=0)  # assign centroid to mean


def reassign_points(centroids, cluster_assignment, dataset, get_distance, k, m):
    """Assign points to new centroids.

    :param centroids:
    :param cluster_assignment:
    :param dataset:
    :param get_distance:
    :param k:
    :param m:
    :return:
    """
    n_count_changed_points = 0
    for i in range(m):  # for each data point assign it to the closest centroid
        min_distance = inf
        min_index = -1
        for j in range(k):
            distance_i_j = get_distance(centroids[j], dataset[i])
            # distance_i_j = get_distance(centroids[j, :], dataset[i, :])
            if distance_i_j < min_distance:
                min_distance = distance_i_j
                min_index = j

        if cluster_assignment[i, 0] != min_index:
            n_count_changed_points += 1

        cluster_assignment[i] = min_index, min_distance ** 2
        # cluster_assignment[i, :] = min_index, min_distance ** 2
    return n_count_changed_points


def should_stop(n_samples, n_count_changed_points):
    """ Judge weather should stop iterate. Using weak stop condition.

    Weak stop condition: The number of changed points less than 1% of n_samples, the number of total points.
    :param cluster_changed: Boolean.
    :param n_samples: int.
    :param n_count_changed_points: int.
    :return: Boolean.
    """
    if n_count_changed_points <= n_samples / 100:
        return True
    return False


def binary_kmeans(dataset, k, get_distance=distance):
    # m = shape(dataset)[0]
    dataset = mat(dataset)
    m = dataset.shape[0]
    cluster_assignment = mat(zeros((m, 2)))
    centroid0 = mean(dataset, axis=0).tolist()[0]
    centroids_list = [centroid0]  # create a list with one centroid
    for j in range(m):  # calc initial Error
        cluster_assignment[j, 1] = get_distance(mat(centroid0), dataset[j, :]) ** 2
    while len(centroids_list) < k:
        lowest_sse = inf
        for i in range(len(centroids_list)):
            points_in_cluster = dataset[nonzero(cluster_assignment[:, 0].A == i)[0],
                                :]  # get the data points currently in cluster i
            centroid_matrix, split_cluster_assignment = kmeans(points_in_cluster, 2, get_distance)
            sse_split = sum(split_cluster_assignment[:, 1])  # compare the SSE to the current minimum
            sse_not_split = sum(cluster_assignment[nonzero(cluster_assignment[:, 0].A != i)[0], 1])
            print "sse_split, and notSplit: ", sse_split, sse_not_split
            if (sse_split + sse_not_split) < lowest_sse:
                best_centroid = i
                best_new_centroid = centroid_matrix
                best_cluster_assignment = split_cluster_assignment.copy()
                lowest_sse = sse_split + sse_not_split
        best_cluster_assignment[nonzero(best_cluster_assignment[:, 0].A == 1)[0], 0] = len(
                centroids_list)  # change 1 to 3,4, or whatever
        best_cluster_assignment[nonzero(best_cluster_assignment[:, 0].A == 0)[0], 0] = best_centroid
        print 'the best_centroid is: ', best_centroid
        print 'the len of best_cluster_assignment is: ', len(best_cluster_assignment)
        centroids_list[best_centroid] = best_new_centroid[0, :].tolist()[
            0]  # replace a centroid with two best centroids
        centroids_list.append(best_new_centroid[1, :].tolist()[0])
        cluster_assignment[nonzero(cluster_assignment[:, 0].A == best_centroid)[0],
        :] = best_cluster_assignment  # reassign new clusters, and SSE
    return mat(centroids_list), cluster_assignment


def load_geography_data(station_address, city):
    api_stem = 'http://where.yahooapis.com/geocode?'  # create a dict and constants for the goecoder
    params = {}
    params['flags'] = 'J'  # JSON return type
    params['appid'] = 'aaa0VN6k'
    params['location'] = '%s %s' % (station_address, city)
    url_params = urllib.urlencode(params)
    yahoo_api = api_stem + url_params  # print url_params
    print yahoo_api
    c = urllib.urlopen(yahoo_api)
    return json.loads(c.read())


def mass_place_find(file_name):
    fw = open('places.txt', 'w')
    for line in open(file_name).readlines():
        line = line.strip()
        line_feature = line.split('\t')
        ret_dict = load_geography_data(line_feature[1], line_feature[2])
        if ret_dict['ResultSet']['Error'] == 0:
            lat = float(ret_dict['ResultSet']['Results'][0]['latitude'])
            lng = float(ret_dict['ResultSet']['Results'][0]['longitude'])
            print "%s\t%f\t%f" % (line_feature[0], lat, lng)
            fw.write('%s\t%f\t%f\n' % (line, lat, lng))
        else:
            print "error fetching"
        sleep(1)
    fw.close()


def SLC_distance(vector_x, vector_y):  # Spherical Law of Cosines
    a = sin(vector_x[0, 1] * pi / 180) * sin(vector_y[0, 1] * pi / 180)
    b = cos(vector_x[0, 1] * pi / 180) * cos(vector_y[0, 1] * pi / 180) * \
        cos(pi * (vector_y[0, 0] - vector_x[0, 0]) / 180)
    return arccos(a + b) * 6371.0  # pi is imported with numpy


def cluster_clubs(n_cluster=5):
    data_list = []
    for line in open('places.txt').readlines():
        line_feature = line.split('\t')
        data_list.append([float(line_feature[4]), float(line_feature[3])])
    data_matrix = mat(data_list)
    my_centroids, clust_assignment = binary_kmeans(data_matrix, n_cluster, get_distance=SLC_distance)
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    scatter_markers = ['s', 'o', '^', '8', 'p',
                       'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    image = plt.imread('Portland.png')
    ax0.imshow(image)
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(n_cluster):
        points_in_cluster = data_matrix[nonzero(clust_assignment[:, 0].A == i)[0], :]
        marker_style = scatter_markers[i % len(scatter_markers)]
        ax1.scatter(points_in_cluster[:, 0].flatten().A[0], points_in_cluster[:, 1].flatten().A[0], marker=marker_style,
                    s=90)
    ax1.scatter(my_centroids[:, 0].flatten().A[0], my_centroids[:, 1].flatten().A[0], marker='+', s=300)
    plt.show()
