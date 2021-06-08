def frequency(labelsInCluster):

    return {i: labelsInCluster.count(i) for i in labelsInCluster}


def calculate_purity(y_pred, y_true):

    purity = 0

    cluster_IDs = set(y_pred)
    majorityLabelFSum = 0

    for cluster_ID in cluster_IDs:

        # for each distinct predicted cluster ID, retrieve the document indices that
        # correspond to that predicted label (which documents belong to which predicted label)
        document_indices = [i for i, j in enumerate(y_pred) if j == cluster_ID]

        # for each document index, retrieve the actual cluster ID for that document to
        # calculate the extent to which a cluster contains a single label
        labelsInCluster = [y_true[i] for i in document_indices]

        # calculate the occurrences of the most frequent label in a cluster
        majorityLabelFSum += max(frequency(labelsInCluster).values())

    purity = majorityLabelFSum / len(y_true)

    return purity
