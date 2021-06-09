import math
from collections import Counter


class InvertedIndex:
    def __init__(self):
        self.index_terms = {}
        self.corpus = Counter()

    def parse_term(self, index_term, document_ID):

        acc = 1

        if document_ID not in self.corpus:
            self.corpus[document_ID] = 0

        if index_term not in self.index_terms:
            self.index_terms[index_term] = Counter()

        self.corpus[document_ID] += acc
        self.index_terms[index_term][document_ID] += acc

    def calculate_term_frequency(self, index_term, document_ID):

        return self.index_terms[index_term].get(document_ID, 0)

    def calculate_document_frequency(self, index_term):

        return len(self.index_terms[index_term])

    def calculate_tf_idf(self, index_term, document_ID):

        tfRaw = self.calculate_term_frequency(index_term, document_ID)

        if not tfRaw == 0:
            tf = 1 + math.log10(tfRaw)
            n = len(self.corpus)
            df = self.calculate_document_frequency(index_term)
            idf = math.log10(n / df)

            return tf * idf

        else:

            return 0

    def make_document_vector(self, document_ID):

        document_vector = []

        for index_term in self.index_terms:

            document_vector.append(self.calculate_tf_idf(index_term, document_ID))

        return document_vector

    def make_document_by_term_matrix(self):

        document_term_matrix = []

        for document_ID in self.corpus:

            document_term_matrix.append(self.make_document_vector(document_ID))

        return document_term_matrix
