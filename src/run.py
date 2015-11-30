import optparse
import os
from random import shuffle, randint
import re
import numpy as np

# parsing
from parsing import *

# features
import sentences
import images

# algorithms
from PyKCCA. kernels import BowKernel, HistKernel
from PyKCCA import KCCA
from sklearn.metrics.pairwise import cosine_similarity

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
    print "Std: ", np.std(rank)
    #print "Var: ", np.var(rank)
    print "Max: ", np.max(rank)

### MAIN ###
def main():
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
    print opts
    print "PARSING################"
    f_train_sen, f_train_img, f_val_sen, f_val_img, f_test_sen, f_test_img = parse_flickr30k_dataset(opts.target, opts.num_flickr_train, 0)
    m_train_sen, m_train_img, m_val_sen, m_val_img, m_test_sen, m_test_img = parse_microsoft_dataset(opts.source, opts.num_microsoft_train, 0, 0)

    print "Training with -> ", "Microsoft: ", len(m_train_sen), len(m_train_img), "Flickr: ", len(f_train_sen), len(f_train_sen)
    print "Testing with -> ", "Microsoft: ", len(m_test_sen), len(m_test_img), "Flickr: ", len(f_test_sen), len(f_test_img)
    train_sen = merge_two_dicts(f_train_sen, m_train_sen)
    train_img = merge_two_dicts(f_train_img, m_train_img)
    test_sen = f_test_sen
    test_img = f_test_img
    print "Parsing complete"

    print "Train sentences", len(train_sen)
    print "Train images", len(train_img)
    print "Test sentences", len(test_sen)
    print "Test images", len(test_img)

    value_sen_train, value_img_train = re_index(train_sen, train_img)
    value_sen_test, value_img_test = re_index(test_sen, test_img)

    if not opts.random:

        #sentences.train_lda(value_sen, 10, opts.out_sentence)
        sentences.train_bow(value_sen_train, opts.out_sentence, opts.out_file)
        images.trainSift(images.PathSet(value_img_train), 256, opts.out_image, opts.out_file)
        print "Training features complete"

        #train_sen_feat = sentences.extract_lda(value_sen, 10, opts.out_sentence)
        train_sen_feat = sentences.extract_bow(value_sen_train, opts.out_sentence, opts.out_file).toarray()

        train_img_feat = images.FeaturesMatrix()
        bad_image_indexes = images.BadIndexes()
        images.extractFeats(opts.out_image, images.PathSet(value_img_train), train_img_feat, bad_image_indexes, opts.out_file)
        train_img_feat = np.asarray(train_img_feat)
        train_sen_feat = np.delete(train_sen_feat, tuple(bad_image_indexes), axis=0)

        print "Extraction of feats complete"
        print "Sentences: ", train_sen_feat.shape
        print "Images: ", train_img_feat.shape

        kernel_sen = BowKernel()
        kernel_img = HistKernel()
        cca = KCCA(kernel_sen, kernel_img,
                   regularization=1e-5,
                   decomp='full',
                   method='kettering_method',
                   scaler1=lambda x:x,
                   scaler2=lambda x:x).fit(train_sen_feat,train_img_feat)

        print "Ftting done",  cca.beta_

        test_sen_feat = sentences.extract_bow(value_sen_test, opts.out_sentence, opts.out_file).toarray()
        test_img_feat = images.FeaturesMatrix()
        bad_image_indexes = images.BadIndexes()
        images.extractFeats(opts.out_image, images.PathSet(value_img_test), test_img_feat, bad_image_indexes, opts.out_file)
        test_img_feat = np.asarray(test_img_feat)
        print "test_sen_feat", test_sen_feat.shape
        print "test_img_feat", test_img_feat.shape

        test_sen_c, test_img_c = cca.transform(test_sen_feat, test_img_feat)
        print "Testing set after transformation"
        print "Sentences: ", test_sen_c.shape
        print "Images: ", test_img_c.shape

        cosine_all = {} # cosine is similarity measure
        pred_index_cosine = []
        print "Predicting... "

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

        print "\nStats"
        print_stats(pred_index_cosine)
        print
        write_to_file(pred_index_cosine, opts.out_file + '_cosine.out')

    else:
        pred_index_random = []
        for i in xrange(0, len(value_img_test)):
            results = [j for j in range(len(value_sen_test))]
            shuffle(results)
            pred_index_random.append(results.index(i))
        print "\nRandom stats"
        print_stats(pred_index_random)
        print
        write_to_file(pred_index_random, opts.out_file + '.out')

if __name__ == "__main__":
    main()
