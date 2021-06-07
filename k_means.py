import numpy as np
from numpy import dot
from numpy.linalg import norm


def cosine_distance(vector1, vector2):

    return 1 - (dot(vector1, vector2) / (norm(vector1) * norm(vector2)))


class KMeans:
    def __init__(self, k, iterations):

        # k is the number of clusters
        self.k = k

        # iterations is one of the termination criteria
        self.iterations = iterations

        # initialize the centroids as an empty list
        self.centroids = []

        # initialize each cluster as an empty list
        self.clusters = [[] for i in range(self.k)]

    def find_nearest_centroid(self, row, centroids):

        # compute cosine distances between a document and the centroids
        cosine_distances = [cosine_distance(row, centroid) for centroid in centroids]

        # retrieve the centroid index with the minimum distance with respect
        # to a row from the TF-IDF matrix
        nearest_centroid_index = np.argmin(cosine_distances)

        return nearest_centroid_index

    def make_clusters(self, centroids):

        # initialize the clusters to a list of lists of length k
        clusters = [[] for i in range(self.k)]

        for document_index, row in enumerate(self.data):

            # find the index of the centroid such that the cosine distance between
            # a row and a centroid is minimized
            centroid_index = self.find_nearest_centroid(row, centroids)

            # append the document index to the cluster that corresponds to the centroid index
            clusters[centroid_index].append(document_index)

        return clusters
