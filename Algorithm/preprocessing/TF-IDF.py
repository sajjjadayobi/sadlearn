import numpy as np
from Algorithm.preprocessing.normalize import l2_norm

"""
TF-IDF.py
detail:  extraction numerical and features of text 

author: sajjad ayobi
see others in repository : sadlearn
in URL: https://github.com/sajjjadayobi/sadlearn/

date : 10/4/2018
"""


class TfidfVectorizer:
    """
    Convert a collection of raw documents to a matrix of TF-IDF features.

    Parameters
    ----------
    docs: get documents to form of list or np.ndarray for convert
    docs sample : ['in method used for natural language processing','...']

    Attributes
    ----------
    words: all words in the documents

    Notes
    -----
    most by import numpy as np
    and from Algorithm.preprocessing.normalize import l2_norm
    l2_norm use for normalization matrix

    Return
    ------
    a matrix than abundance words in texts

    Use
    ---
    tf = TfidfVectorizer(documents)
    matrix = tf.transform()
    """

    def __init__(self, docs):
        self.docs = docs
        words = self.extract_words(docs)

        self.words = self.remove_surplus_chars(words)

    @staticmethod
    def remove_surplus_chars(words):
        added_words = ['a', '.', '(', ')', ',', '$', ' ', '?', '!']

        words = np.unique(np.sort(words))
        for i, word in enumerate(words):
            if word in added_words:
                words = np.delete(words, i)
        return words

    @staticmethod
    def extract_words(docs):
        words = np.empty(0)
        for doc in docs:
            words = np.append(words, doc.lower().split(' '))
        return words

    def counts_words(self, doc):
        doc = doc.split(' ')
        counts = []
        for word in self.words:
            count = 0
            for i in doc:
                if i == word:
                    count += 1
            counts.append(count)
        return counts

    def transform(self):

        arr = np.zeros((len(self.docs), len(self.words)))
        for i, doc in enumerate(self.docs):
            arr[i] = self.counts_words(doc)

        transformed = l2_norm(arr)
        return transformed
