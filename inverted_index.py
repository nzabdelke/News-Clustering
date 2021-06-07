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
