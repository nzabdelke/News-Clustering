import numpy as np
from numpy import dot
from numpy.linalg import norm


def cosine_distance(vector1, vector2):

    return 1 - (dot(vector1, vector2) / (norm(vector1) * norm(vector2)))
