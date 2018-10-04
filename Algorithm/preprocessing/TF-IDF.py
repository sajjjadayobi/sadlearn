import numpy as np
from Algorithm.preprocessing.normalize import l2_norm

docs = ["machine learning is a road for artificial inelegance and this advanced is deep learning",
        "python is a best programing language for artificial inelegance and you can ues",
        "can javascript language developed frontend website and support by facebook company",
        'you are watching machine learning course',
        'word frequency array is apart of unsupervised learning from machine learning course',
        'corsera is an online educational website']


def remove_added_words(words):
    added_words = ['a', '.', '(', ')', ',', '$', ' ', '?', '!']

    words = np.unique(np.sort(words))
    for i, word in enumerate(words):
        if word in added_words:
            words = np.delete(words, i)
    return words


def get_all_words(docs):
    words = np.empty(0)
    for doc in docs:
        words = np.append(words, doc.lower().split(' '))
    words = remove_added_words(words)
    return words


def count_words(doc, words):
    doc = doc.split(' ')
    counts = []
    for word in words:
        count = 0
        for i in doc:
            if i == word:
                count += 1
        counts.append(count)
    return counts


def tf_idf(docs):
    words = get_all_words(docs)
    arr = np.zeros((len(docs), len(words)))
    for i, doc in enumerate(docs):
        arr[i] = count_words(doc, words)

    return l2_norm(arr)


print(tf_idf(docs))
