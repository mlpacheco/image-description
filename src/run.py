import optparse
import os
from random import shuffle

import nn

def get_splits(total, train_ratio, val_ratio):
    num_train = int(total*train_ratio)
    num_val = int(total*val_ratio)
    return num_train, num_val

def get_split_ids_microsoft(train_ratio, val_ratio):
    num_train, num_val = get_splits(10200, train_ratio, val_ratio)
    ids = [i for i in xrange(0, 10200)]
    shuffle(ids)
    return set(ids[:num_train]),\
           set(ids[num_train:num_train+num_val])

def parse_microsoft_sentences(domain, train_ratio, val_ratio):
    train_ids, val_ids = get_split_ids_microsoft(train_ratio, val_ratio)
    filename1 = os.path.join(domain, "SimpleSentences", "SimpleSentences1_10020.txt")
    filename2 = os.path.join(domain, "SimpleSentences", "SimpleSentences2_10020.txt")
    train_data = {}; val_data = {}; test_data = {}
    for filename in [filename1, filename2]:
        with open(filename) as f:
            f = f.readlines()
            for s in f:
                s = s.strip().split("\t", 2)
                index = int(s[0])
                if index in train_ids:
                    sentences = train_data
                elif index in val_ids:
                    sentences = val_data
                else:
                    sentences = test_data

                if index in sentences:
                    sentences[index] += " " + s[2]
                else:
                    sentences[index] = s[2]
    return train_data, val_data, test_data

def parse_flickr30k_sentences(domain):
    pass

def parse_microsoft_images(domain):
    pass

def parse_flickr30k_images(domain):
    pass


def main():
    parser = optparse.OptionParser()
    parser.add_option('-s', '--source', help='path to source domain dir',
                      dest='source', type='string')
    parser.add_option('-t', '--target', help='path to target domain dir',
                      dest='target', type='string')
    parser.add_option('-o', '--out', help='path to output file for train instances',
                      dest='output', type='string')
    parser.add_option('--rt', help='ratio of examples for training',
                      dest='train_ratio', type='int', default=0.7)
    parser.add_option('--rv', help='ratio of examples for validation',
                      dest='val_ratio', type='int', default=0.0)
    (opts, args) = parser.parse_args()

    train_ss, val_ss, test_ss = parse_microsoft_sentences(opts.source,
                                                          opts.train_ratio,
                                                          opts.val_ratio)
    sentence_feat = nn.train_sentence_features(train_ss)
    nn.save_sentence_instances(sentence_feat, opts.output)

    test_index = test_ss.keys()[0]
    print nn.max_sentence_similarity(test_ss[test_index], train_ss,
                                     sentence_feat, len(train_ss))


if __name__ == "__main__":
    main()
