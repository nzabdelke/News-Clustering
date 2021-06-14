import numpy as np
from numpy.linalg import eig


class PCA:
    def __init__(self, num_of_components):
        self.num_of_components = num_of_components
        self.components = None

    def fit(self, X):

        # calculate the mean
        mean = np.mean(X, axis=0)

        # calculate the covariance matrix
        covariance_matrix = np.cov((X - mean).T)

        # calculate the eigenvectors and their corresponding eigenvalues
        eigenvalues, eigenvectors = eig(covariance_matrix)

        # transpose eigenvectors from column vectors to row vectors
        eigenvectors = eigenvectors.T

        # find the indices from sorting the eigenvalues in descending order
        sorted_eigenvalues_indices = np.argsort(eigenvalues)[::-1]

        # sort the eigenvectors according to their corresponding eigenvalues
        eigenvectors = eigenvectors[sorted_eigenvalues_indices]

        # store the first num_of_components eigenvectors
        self.components = eigenvectors[: self.num_of_components]

    def transform(self, X):
        pass
