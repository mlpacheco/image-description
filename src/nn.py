import optparse
import os
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from sentences import f1_similarity

def train_sentence_features(sentences):
    vectorizer = CountVectorizer(binary=True)
    freq = vectorizer.fit_transform(sentences.values())
    sums = np.asarray(freq.sum(axis=0)).ravel()
    words = vectorizer.get_feature_names()
    dictionary = dict(zip(words, sums))
    return dictionary

def save_sentence_instances(dictionary, filename):
    with open(filename, 'w') as outfile:
        json.dump(dictionary, outfile)


def max_sentence_similarity(s_query, sentences, dictionary, train_size):
    max_sim = 0.0
    for i,s in sentences.iteritems():
        curr_sim = f1_similarity(s_query, s, dictionary, train_size)
        if curr_sim > max_sim:
            max_sim = curr_sim
            nn_img = i
    return nn_img, max_sim

def max_image_similarity(i_query, images):
    # find image closest to i_query and return sentence
    return

def score_image():
    return

def score_sentence():
    return
