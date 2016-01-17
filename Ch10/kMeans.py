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
from numpy import *


# TODO: optimize numpy import

def load_dataset(file_name):  # general function to parse tab -delimited floats
    data_matrix = []  # assume last column is target value
    fr = open(file_name)
    for line in fr.readlines():
        current_line = line.strip().split('\t')
        float_line = map(float, current_line)  # map all elements to float()
        data_matrix.append(float_line)
    return data_matrix


def distance(vector_x, vector_b):
    return sqrt(sum(power(vector_x - vector_b, 2)))  # la.norm(vector_x-vector_b)


def get_centroids(dataset, k):
    n = shape(dataset)[1]
    centroids = mat(zeros((k, n)))  # create centroid mat
    for j in range(n):  # create random cluster centers, within bounds of each dimension
        min_j = min(dataset[:, j])
        range_j = float(max(dataset[:, j]) - min_j)
        centroids[:, j] = mat(min_j + range_j * random.rand(k, 1))
    return centroids


def kmeans(dataset, k, get_distance=distance, create_centroids=get_centroids):
    m = shape(dataset)[0]
    cluster_assignment = mat(zeros((m, 2)))  # create mat to assign data points
    # to a centroid, also holds SE of each point
    centroids = create_centroids(dataset, k)
    cluster_changed = True
    while cluster_changed:
        cluster_changed = False
        for i in range(m):  # for each data point assign it to the closest centroid
            min_distance = inf
            min_index = -1
            for j in range(k):
                distance_i_j = get_distance(centroids[j, :], dataset[i, :])
                if distance_i_j < min_distance:
                    min_distance = distance_i_j
                    min_index = j
            if cluster_assignment[i, 0] != min_index: cluster_changed = True
            cluster_assignment[i, :] = min_index, min_distance ** 2
        print centroids
        for cent in range(k):  # recalculate centroids
            points_in_cluster = dataset[
                nonzero(cluster_assignment[:, 0].A == cent)[0]]  # get all the point in this cluster
            centroids[cent, :] = mean(points_in_cluster, axis=0)  # assign centroid to mean
    return centroids, cluster_assignment


def binary_kmeans(dataset, k, get_distance=distance):
    m = shape(dataset)[0]
    cluster_assignment = mat(zeros((m, 2)))
    centroid0 = mean(dataset, axis=0).tolist()[0]
    centroids_list = [centroid0]  # create a list with one centroid
    for j in range(m):  # calc initial Error
        cluster_assignment[j, 1] = get_distance(mat(centroid0), dataset[j, :]) ** 2
    while len(centroids_list) < k:
        lowest_SSE = inf
        for i in range(len(centroids_list)):
            points_in_cluster = dataset[nonzero(cluster_assignment[:, 0].A == i)[0],
                                :]  # get the data points currently in cluster i
            centroid_matrix, split_cluster_assignment = kmeans(points_in_cluster, 2, get_distance)
            sse_split = sum(split_cluster_assignment[:, 1])  # compare the SSE to the currrent minimum
            sse_not_split = sum(cluster_assignment[nonzero(cluster_assignment[:, 0].A != i)[0], 1])
            print "sse_split, and notSplit: ", sse_split, sse_not_split
            if (sse_split + sse_not_split) < lowest_SSE:
                best_centroid = i
                best_new_centroid = centroid_matrix
                best_cluster_assignment = split_cluster_assignment.copy()
                lowest_SSE = sse_split + sse_not_split
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


def load_geography_data(stAddress, city):
    apiStem = 'http://where.yahooapis.com/geocode?'  # create a dict and constants for the goecoder
    params = {}
    params['flags'] = 'J'  # JSON return type
    params['appid'] = 'aaa0VN6k'
    params['location'] = '%s %s' % (stAddress, city)
    url_params = urllib.urlencode(params)
    yahooApi = apiStem + url_params  # print url_params
    print yahooApi
    c = urllib.urlopen(yahooApi)
    return json.loads(c.read())


def mass_place_find(fileName):
    fw = open('places.txt', 'w')
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        retDict = load_geography_data(lineArr[1], lineArr[2])
        if retDict['ResultSet']['Error'] == 0:
            lat = float(retDict['ResultSet']['Results'][0]['latitude'])
            lng = float(retDict['ResultSet']['Results'][0]['longitude'])
            print "%s\t%f\t%f" % (lineArr[0], lat, lng)
            fw.write('%s\t%f\t%f\n' % (line, lat, lng))
        else:
            print "error fetching"
        sleep(1)
    fw.close()


def SLC_distance(vecA, vecB):  # Spherical Law of Cosines
    a = sin(vecA[0, 1] * pi / 180) * sin(vecB[0, 1] * pi / 180)
    b = cos(vecA[0, 1] * pi / 180) * cos(vecB[0, 1] * pi / 180) * \
        cos(pi * (vecB[0, 0] - vecA[0, 0]) / 180)
    return arccos(a + b) * 6371.0  # pi is imported with numpy


def cluster_clubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)
    myCentroids, clustAssing = binary_kmeans(datMat, numClust, get_distance=SLC_distance)
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    scatterMarkers = ['s', 'o', '^', '8', 'p', \
                      'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:, 0].A == i)[0], :]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0], ptsInCurrCluster[:, 1].flatten().A[0], marker=markerStyle,
                    s=90)
    ax1.scatter(myCentroids[:, 0].flatten().A[0], myCentroids[:, 1].flatten().A[0], marker='+', s=300)
    plt.show()
