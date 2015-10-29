import optparse
import os
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def parse_microsoft_sentences(domain):
    filename1 = os.path.join(domain, "SimpleSentences", "SimpleSentences1_10020.txt")
    filename2 = os.path.join(domain, "SimpleSentences", "SimpleSentences2_10020.txt")
    sentences = {}
    for filename in [filename1, filename2]:
        with open(filename) as f:
            f = f.readlines()
            for s in f:
                s = s.strip().split("\t", 2)
                if s[0] in sentences:
                    sentences[s[0]] += " " + s[2]
                else:
                    sentences[s[0]] = s[2]

    return sentences


def parse_flickr30k_sentences(domain):
    pass

def parse_microsoft_images(domain):
    pass

def parse_flickr30k_images(domain):
    pass

def train_sentence_features(sentences):
    vectorizer = CountVectorizer()
    freq = vectorizer.fit_transform(sentences.values())
    sums = np.asarray(freq.sum(axis=0)).ravel()
    words = vectorizer.get_feature_names()
    dictionary = dict(zip(words, sums))
    return dictionary

def save_sentence_features(dictionary, filename):
    with open(filename, 'w') as outfile:
        json.dump(dictionary, outfile)

def main():
    parser = optparse.OptionParser()
    parser.add_option('-s', '--source', help='path to source domain dir',
                      dest='source', type='string')
    parser.add_option('-t', '--target', help='path to target domain dir',
                      dest='target', type='string')
    parser.add_option('-o', '--out', help='path to output dir for train instances',
                      dest='output', type='string')
    (opts, args) = parser.parse_args()

    source_sent = parse_microsoft_sentences(opts.source)
    feat_dict = train_sentence_features(source_sent)
    save_sentence_features(feat_dict, opts.output)

    return

if __name__ == "__main__":
    main()
