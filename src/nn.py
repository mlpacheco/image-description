import optparse
import os
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from sentences import f1_similarity
import images

def train_sentence_features(sentences, filename):
    vectorizer = CountVectorizer(binary=True)
    freq = vectorizer.fit_transform(sentences.values())
    sums = np.asarray(freq.sum(axis=0)).ravel()
    words = vectorizer.get_feature_names()
    dictionary = dict(zip(words, sums))

    with open(filename, 'w') as outfile:
        json.dump(dictionary, outfile)

    return filename

def train_image_features(image_list, num_words, filename):
    print type(image_list.values())
    imagePaths = images.PathSet(image_list.values())
    images.trainSift(imagePaths, num_words, filename)
    return filename

def max_sentence_similarity(s_query, sentences, filename, train_size):
    with open(filename) as data_file:
        dictionary = json.load(data_file)

    max_sim = 0.0
    for i,s in sentences.iteritems():
        curr_sim = f1_similarity(s_query, s, dictionary, train_size)
        if curr_sim > max_sim:
            max_sim = curr_sim
            nn_img = i
    return nn_img, max_sim

def max_image_similarity(i_query, image_list, filename):
    max_sim = 0.0
    for i,s in image_list.iteritems():
        curr_sim = images.similarityScore(i_query, i, filename)
        if curr_sim > max_sim:
            max_sim = curr_sim
            nn_sent = s
    return nn_sent, max_sim

def score_image():
    return

def score_sentence():
    return
