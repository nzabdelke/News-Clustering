import numpy as np
from tabulate import tabulate

from clustering.external_evaluation import calculate_purity
from clustering.k_means import KMeans
from data_preparation.inverted_index import InvertedIndex
from data_preparation.pre_processing import parse_corpus, pre_process_corpus

corpus, y_true, titles = parse_corpus()
preprocessed_corpus = pre_process_corpus(corpus)

y_true = [y_true.index(l) for l in y_true]


def generate_matrix(preprocessed_corpus):
    inverted_index = InvertedIndex()

    for i in range(len(preprocessed_corpus)):
        for term in preprocessed_corpus[i].split():
            inverted_index.parse_term(term, i)
    document_term_matrix = np.array(inverted_index.make_document_by_term_matrix())
    return document_term_matrix


matrix = generate_matrix(preprocessed_corpus)
k = KMeans(5, 1000)

document_clusters = k.assign_documents_to_cluster(matrix)

y_pred = document_clusters[0]


clusters = document_clusters[1]

cluster_tightness = document_clusters[2]


top_documents = document_clusters[3]


def write_clusters():
    with open("clusters.txt", "w") as f:
        for i in range(len(clusters)):
            data = []

            f.write(
                "Cluster #%d contains the following %d documents: "
                % (i, len(clusters[i]))
            )
            f.write("\n\n")
            for j in range(len(clusters[i])):
                id = clusters[i][j]
                data.append([id, titles[id]])
            f.write(tabulate(data, headers=["Document ID", "Document Title"]))
            f.write("\n\n")


def sort_tuples(tuples):

    # sort tuples in ascending order by the second element
    # (distance from the centroid), which acts as the key

    tuples.sort(key=lambda x: x[1])
    return tuples


def show_summary():
    for i in range(len(top_documents)):
        data = []
        print("The top 3 documents in cluster #%d are:\n " % i)
        sortedTuples = sort_tuples(top_documents[i])[:3]
        for j in sortedTuples:
            data.append([j[0], titles[j[0]]])
        print(tabulate(data, headers=["Document ID", "Document Title"]))
        print()


def show_RSS():
    data = []
    for i in range(len(cluster_tightness)):
        data.append([i, cluster_tightness[i]])

    print(tabulate(data, headers=["Cluster ID", "RSS"]))

    print("\nThe total RSS is %.2f." % sum(cluster_tightness))


def show_purity():
    purity = calculate_purity(y_pred, y_true)
    print("The purity is %.2f." % (100 * purity))


def display_menu():

    # display menu shown to user
    print("")
    print(60 * "-", "Menu", 60 * "-")
    print("1. Show Cluster Summary")
    print("2. Calculate RSS")
    print("3. Calculate Purity")
    print("4. Write Clusters")
    print("5. Exit")
    print(127 * "-")
    print("")


def wait_for_input():
    input("\nPlease press Enter to continue...")


status = True

# main loop to display the menu
while status:
    display_menu()
    selection = input("Please enter your selection (1-4): ")
    print()
    if selection == "1":
        show_summary()
        wait_for_input()

    elif selection == "2":
        show_RSS()
        wait_for_input()

    elif selection == "3":
        show_purity()
        wait_for_input()

    elif selection == "4":
        write_clusters()
        wait_for_input()

    elif selection == "5":
        print("\nThe program will now terminate.")
        status = False

    else:

        # prompt user for a valid selection
        input("Please select a valid option from the menu.\n")
