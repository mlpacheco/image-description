from math import log

def f1_similarity(s1, s2):
    common = sum([word_idf(w) for w in common_words(s1, s2)])
    x = sum([word_idf(w) for w in s1])
    y = sum([word_idf(w) for w in s2])
    f1 = 1 - ((2.0*common)/(x + y))
    return f1

def word_idf(word_freq, train_size):
    idf = float(train_size) / (word_freq + 1)
    return log(idf)

def common_words(s1, s2):
    words = set(s1) & set(s2)
    return list(words)

