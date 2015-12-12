import optparse
import string
import os
import re

from nltk.stem.wordnet import WordNetLemmatizer
from practnlptools.tools import Annotator

def preprocess_flickr(domain):
    filename1 = os.path.join(domain, "train_sentences.token")
    filename2 = os.path.join(domain, "test_sentences.token")
    outname1 = os.path.join(domain, "train_sentences_preproc.token")
    outname2 = os.path.join(domain, "test_sentences_preproc.token")
    posname1 = os.path.join(domain, "train_sentences_pos.token")
    posname2 = os.path.join(domain, "test_sentences_pos.token")

    preprocess(filename1, outname1, posname1, 1)
    preprocess(filename2, outname2, posname2, 1)

def preprocess_microsoft(domain):
    filename1 = os.path.join(domain, "SimpleSentences", "newSimpleSentences1_10020.txt")
    filename2 = os.path.join(domain, "SimpleSentences", "newSimpleSentences2_10020.txt")
    outname1 = os.path.join(domain, "SimpleSentences", "newSimpleSentences1_10020_preproc.txt")
    outname2 = os.path.join(domain, "SimpleSentences", "newSimpleSentences2_10020_preproc.txt")
    posname1 = os.path.join(domain, "SimpleSentences", "newSimpleSentences1_10020_pos.txt")
    posname2 = os.path.join(domain, "SimpleSentences", "newSimpleSentences2_10020_pos.txt")

    #preprocess(filename1, outname1, posname1, 2)
    preprocess(filename2, outname2, posname2, 2)

def preprocess(infile, outfile, posfile, index):
    annotator = Annotator()
    wnl = WordNetLemmatizer()
    o = open(outfile, 'w'); p = open(posfile, 'w'); f = open(infile)
    text = f.readlines()
    for s in text:
        s = s.strip().split("\t", index)
        # make it lower
        sent = s[index].lower()
        # remove special characters
        sent = sent.strip(string.punctuation)
        # extend contractions
        sent = re.sub(r"n't", " not", sent)
        sent = re.sub(r"'ve", " have", sent)
        sent = re.sub(r"'d", " would", sent)
        sent = re.sub(r"'ll", " will", sent)
        sent = re.sub(r"'m", " am", sent)
        sent = re.sub(r"'s", " is", sent)
        sent = re.sub(r"'re", " are", sent)

        # lematize and get POS tags
        pos = annotator.getAnnotations(sent)["pos"]
        lemmas = [wnl.lemmatize(w,'v') if t.startswith('V') else wnl.lemmatize(w, 'n') for (w,t) in pos]
        sent = " ".join(lemmas)
        pos = " ".join([x[1] for x in pos])

        out_string = ""
        pos_string = ""
        for j in range(0,index):
            out_string += s[j] + "\t"
            pos_string += s[j] + "\t"

        out_string += sent + "\n"
        pos_string += pos + "\n"
        o.write(out_string)
        p.write(pos_string)

    f.close()
    o.close()
    o.close()


parser = optparse.OptionParser()
parser.add_option('-p', '--path', help='path to the sentences files', type='string')
parser.add_option('-d', '--domain', help='0: microsoft, 1: flickr', type='int')
(opts, args) = parser.parse_args()

if opts.domain == 0:
    preprocess_microsoft(opts.path)
elif opts.domain == 1:
    preprocess_flickr(opts.path)
else:
    parser.print_help()


