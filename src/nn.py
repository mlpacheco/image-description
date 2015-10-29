import optparse
import os
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def train_sentence_features(sentences):
    vectorizer = CountVectorizer()
    freq = vectorizer.fit_transform(sentences.values())
    sums = np.asarray(freq.sum(axis=0)).ravel()
    words = vectorizer.get_feature_names()
    dictionary = dict(zip(words, sums))
    return dictionary

def save_sentence_instances(dictionary, filename):
    with open(filename, 'w') as outfile:
        json.dump(dictionary, outfile)

