import numpy as np
from numpy import dot
from numpy.linalg import norm

np.random.seed(123)


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

    def cluster_labels(self, clusters):

        # initialize the labels as an empty 1D array with a length corresponding
        # to the number of documents in the TF-IDF matrix
        labels = np.empty(self.number_of_documents)

        for cluster_index, cluster in enumerate(clusters):

            for document_index in cluster:

                # the label of a document is assigned the index of its cluster
                labels[document_index] = cluster_index

        return labels

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

        for document_index, row in enumerate(self.instance_feature_matrix):

            # find the index of the centroid such that the cosine distance between
            # a row and a centroid is minimized
            centroid_index = self.find_nearest_centroid(row, centroids)

            # append the document index to the cluster that corresponds to the centroid index
            clusters[centroid_index].append(document_index)

        return clusters

    def calculate_RSS(self, clusters, centroids):

        # store RSS of each cluster
        RSS_list = []

        top_documents = [[] for y in range(self.k)]

        squared_distance = 0

        for cluster_index, document_indices in enumerate(clusters):

            cluster_RSS = []

            for document_index in document_indices:

                # compute the distance squared of a document with respect to its centroid
                squared_distance = (
                    cosine_distance(
                        self.instance_feature_matrix[document_index],
                        centroids[cluster_index],
                    )
                    ** 2
                )

                cluster_RSS.append(squared_distance)

                # store the documents that have a distance less than
                # 0.8 with respect to the document's cluster centroid
                distance = np.sqrt(squared_distance)

                if distance < 0.8:
                    top_documents[cluster_index].append((document_index, distance))

            # RSS of a cluster is the sum of squared distances between document
            # vectors and their respective centroid
            RSS_list.append(sum(cluster_RSS))

        return (RSS_list, top_documents)

    def calculate_centroids(self, clusters):

        # initialize centroids as nD array of size k * number of index terms
        centroids = np.zeros((self.k, self.number_of_index_terms))

        for cluster_index, document_indices in enumerate(clusters):

            # assign the cluster mean to the corresponding centroid
            centroids[cluster_index] = np.mean(
                self.instance_feature_matrix[document_indices], axis=0
            )

        return centroids

    def is_converged(self, prev_centroids, centroids):

        # for each cluster, calculate the cosine distance between the
        # current centroid and the previous centroid
        cosine_distances = [
            cosine_distance(prev_centroids[i], centroids[i]) for i in range(self.k)
        ]

        converged = sum(cosine_distances) == 0
        # convergance occurs when there is no change in the cosine_distance between
        # the previous centroid and the current centroid
        return converged

    def assign_documents_to_cluster(self, instance_feature_matrix):

        self.instance_feature_matrix = instance_feature_matrix

        # number of documents is the number of rows in the TF-IDF matrix
        self.number_of_documents = instance_feature_matrix.shape[0]

        # number of index terms is the number of columns in the TF-IDF matrix
        self.number_of_index_terms = instance_feature_matrix.shape[1]

        # retrieve the document indices that will serve as the seed centroids randomly
        seed_document_indices = np.random.choice(
            self.number_of_documents, self.k, replace=False
        )  # replace = False to sample without replacement

        # assign the centroids to the actual row from the TF-IDF matrix that corresponds to the
        # randomly selected document indices
        self.centroids = [
            self.instance_feature_matrix[i] for i in seed_document_indices
        ]

        # update clusters and centroids
        for i in range(self.iterations):

            # update clusters
            self.clusters = self.make_clusters(self.centroids)

            prev_centroids = self.centroids

            # for each cluster, assign the mean value of the cluster to the corresponding
            # centroid
            self.centroids = self.calculate_centroids(self.clusters)

            # check for convergence and exit if this occurs before the number of iterations is reached
            if self.is_converged(prev_centroids, self.centroids):
                break

        RSS, top_documents = self.calculate_RSS(self.clusters, self.centroids)

        # return the cluster label for every document and other variables used for evaluation / cluster understanding
        return (self.cluster_labels(self.clusters), self.clusters, RSS, top_documents)
