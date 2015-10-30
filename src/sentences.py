from math import log

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

