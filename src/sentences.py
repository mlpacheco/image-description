from math import log
#from gensim import corpora, models
from os.path import join
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
import numpy as np

# Default filenames
LDA_FILENAME = "lda.pkl"
DICT_FILENAME = "dictionary.txt"
BOW_FILENAME = "BOWvectorizer.pkl"
SEL_FILENAME = "lowVarSelector.pkl"

# similarity measure proposed by hockenmeier on
# the NN approach, unable to use it on
# the frustratingly easy framework
def f1_similarity(s1, s2, dictionary, train_size):
    common = sum([word_idf(dictionary[w], train_size)\
                  if w in dictionary else 0.0\
                  for w in common_words(s1, s2)])
    x = sum([word_idf(dictionary[w], train_size) \
             if w in dictionary else 0.0 for w in s1.split()])
    y = sum([word_idf(dictionary[w], train_size)\
             if w in dictionary else 0.0 for w in s2.split()])
    if (x + y) == 0:
        return 0.0
    else:
        return 1 - ((2.0*common)/(x + y))

def word_idf(word_freq, train_size):
    idf = float(train_size) / (word_freq + 1)
    return log(idf)

def common_words(s1, s2):
    words = set(s1.split()) & set(s2.split())
    return list(words)

# k: number of topics to extract
def train_lda(sentences, k, path, out_file):
    dict_filename = join(path, out_file + "_dict.txt")
    model_filename = join(path, out_file + "_lda.pkl")
    documents = [x.split() for x in sentences]
    dictionary = corpora.Dictionary(documents)
    # filter extremes?
    dictionary.filter_extremes(no_below=10)
    corpus = map(lambda x: dictionary.doc2bow(x), documents)
    model = models.LdaModel(corpus, num_topics=k, id2word=dictionary, iterations=1000)
    topics = model.show_topics(num_topics=k, num_words=5)

    #for topic in topics:
    #    print topic

    ## save trained dictionary and model in files
    dictionary.save_as_text(dict_filename)
    model.save(model_filename)

def extract_lda(sentences, k, path, out_file):
    dict_filename = join(path, out_file + "_dict.txt")
    model_filename = join(path, out_file + "_lda.pkl")
    dictionary = corpora.Dictionary.load_from_text(dict_filename)
    model = models.LdaModel.load(model_filename)

    documents = [x.split() for x in sentences]

    ret = []
    for d in documents:
        feats = [0.0]*k
        topics = model[dictionary.doc2bow(d)]
        for (i,p) in topics:
            feats[i] = p
        ret.append(feats)
    return np.asarray(ret)


def train_bow(sentences, path, out_file):
    ## use of TF-IDF normalization for BOW
    tfidf_filename = join(path, out_file + "_bow.pkl")
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, min_df=0.01)
    vectorizer.fit(sentences)
    joblib.dump(vectorizer, tfidf_filename)

def extract_bow(sentences, path, out_file):
    tfidf_filename = join(path, out_file + "_bow.pkl")
    vectorizer = joblib.load(tfidf_filename)
    return vectorizer.transform(sentences)
