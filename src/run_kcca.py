import optparse
import os
from random import shuffle, randint
import re
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from PyKCCA import KCCA
from parsing import *
import sentences
import images
from kernels import BowKernel, HistKernel

def parse_input():
    parser = optparse.OptionParser()
    parser.add_option('-f', '--flickr', help='number of training examples of target domain',\
                      dest='num_flickr_train', type='int')
    parser.add_option('-s', '--source', help='path to source domain dir',\
                      dest='source', type='string')
    parser.add_option('-t', '--target', help='path to target domain dir',\
                      dest='target', type='string')
    parser.add_option('-w', '--words', help='output path for sentence files',\
                      dest='out_sentence', type='string')
    parser.add_option('-i', '--images', help='output path for image files',\
                      dest='out_image', type='string')
    parser.add_option('-m', '--microsoft', help='number of training examples of source domain',\
                      dest='num_microsoft_train', type='int')
    parser.add_option('-o', '--out', help='output file', dest='out_file', type='string')
    parser.add_option('-r', '--random', help='random ranking', dest='random', action='store_true')
    (opts, args) = parser.parse_args()
    '''mandatories = ['source', 'target', 'out_image', 'out_sentence', 'num_microsoft_train', 'num_flickr_train', 'out_file']
    for m in mandatories:
        if not opts.__dict__[m]:
            print m
            print "mandatory option is missing\n"
            parser.print_help()
            exit(i-1)'''
    return opts

def parse_datasets(opts):
    f_train_sen, f_train_img, f_val_sen, f_val_img, f_test_sen, f_test_img = parse_flickr30k_dataset(opts.target, opts.num_flickr_train, 0)
    m_train_sen, m_train_img, m_val_sen, m_val_img, m_test_sen, m_test_img = parse_microsoft_dataset(opts.source, opts.num_microsoft_train, 0, 0)
    train_sen = merge_two_dicts(f_train_sen, m_train_sen)
    train_img = merge_two_dicts(f_train_img, m_train_img)
    test_sen = f_test_sen
    test_img = f_test_img

    value_sen_train, value_img_train = re_index(train_sen, train_img)
    value_sen_test, value_img_test = re_index(test_sen, test_img)
    return value_sen_train, value_img_train, value_sen_test, value_img_test

def write_to_file(scores, filename):
    with open(filename, 'w') as fw:
        for i in scores:
            fw.write(str(i))
            fw.write('\n')
        fw.close()

def print_stats(rank):
    rank = np.asarray(rank)
    print "Min: ", np.min(rank)
    print "Avg: ", np.mean(rank)
    print "Med: ", np.median(rank)
    print "Std: ", np.std(rank)
    print "Max: ", np.max(rank)

def rank_retrieval(test_sen_c, test_img_c, outfile):
    cosine_all = {} # cosine is similarity measure
    pred_index_cosine = []
    for i in xrange(0, len(test_sen_c)):
            X = test_sen_c[i]
            l1 = []; l2 = []; cosine = [];
            for j in xrange(0, len(test_sen_c)):
                Y = test_img_c[j]
                cosine.append((cosine_similarity(X.reshape(1, -1), Y.reshape(1, -1))[0][0], j))

            cosine.sort(reverse=1)
            #print cosine
            cosine_all[i] = [index for (metric,index) in cosine]
            pred_index_cosine.append(cosine_all[i].index(i))

    print "\nCosine Stats"
    print_stats(pred_index_cosine)
    print
    write_to_file(pred_index_cosine, outfile + '_cosine.out')

def rank_random(value_sen_test, value_img_test, outfile):
    pred_index_random = []
    for i in xrange(0, len(value_img_test)):
        results = [j for j in range(len(value_sen_test))]
        shuffle(results)
        pred_index_random.append(results.index(i))

    print "\nRandom Stats"
    print_stats(pred_index_random)
    print
    write_to_file(pred_index_random, outfile + '.out')


def main():
    opts = parse_input()
    print opts

    print "PARSING ################"
    value_sen_train, value_img_train, value_sen_test, value_img_test = parse_datasets(opts)
    print "Done."

    if not opts.random:

        print "TRAINING FEATURES ##############"
        sentences.train_bow(value_sen_train, opts.out_sentence, opts.out_file)
        images.trainSift(images.PathSet(value_img_train), 256, opts.out_image, opts.out_file)
        images.trainCielab(images.PathSet(value_img_train), 128, opts.out_image, opts.out_file)
        print "Done."

        kernel_sen = BowKernel()
        kernel_img = HistKernel(opts.out_image, opts.out_file)

        print "FITTING KCCA ##################"
        value_sen_train = sentences.extract_bow(value_sen_train, opts.out_sentence, opts.out_file).toarray()
        value_sen_test = sentences.extract_bow(value_sen_test, opts.out_sentence, opts.out_file).toarray()

        # image kernel needs to come first on all KCCA calls for bad images error handling
        cca = KCCA(kernel_img, kernel_sen,
                       regularization=1e-5,
                       decomp='full',
                       method='kettering_method',
                       scaler1=lambda x:x,
                       scaler2=lambda x:x).fit(value_img_train, value_sen_train)

        print "Done",  cca.beta_

        print "KCCA TRANSFORM ##################"
        test_img_c, test_sen_c = cca.transform(value_img_test, value_sen_test)
        print "Sentences: ", test_sen_c.shape
        print "Images: ", test_img_c.shape

        print "RANKING #############"
        rank_retrieval(test_sen_c, test_img_c, opts.out_file)

    else:
        rank_random(value_sen_test, value_img_test, opts.out_file)

if __name__ == "__main__":
    main()
