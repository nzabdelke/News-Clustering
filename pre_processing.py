import glob
import os

import Stemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer


def parse_corpus():

    titles = []
    documents = []
    y_true = []

    for label in ["athletics", "cricket", "football", "rugby", "tennis"]:

        for file in glob.glob(os.path.join("bbcsport", label, "*")):

            curr_file = open(file, encoding="utf8", errors="ignore")

            # store each document and title as a string for input
            # to the inverted index
            documents.append(curr_file.read())

            # store the natural label of each document
            y_true.append(label)

            # go back to the top of a file to obtain the document title
            curr_file.seek(0)
            titles.append(curr_file.readline().strip())

    return (documents, titles, y_true)


def pre_process_corpus(documents):

    stemmer = Stemmer.Stemmer("english")
    stopwords_set = set(stopwords.words("english"))
    my_tokenizer = CountVectorizer().build_tokenizer()
    pre_processed_corpus = []

    for document in documents:

        pre_processed_corpus.append(
            " ".join(
                [
                    stemmer.stemWord(i.lower())
                    for i in my_tokenizer(document)
                    if i.lower() not in stopwords_set
                ]
            )
        )

    return pre_processed_corpus
