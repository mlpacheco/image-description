import optparse
import os
from random import shuffle, randint
import re
import numpy as np

# features
import sentences
import images

# algorithms
#from sklearn.cross_decomposition import CCA
#import rcca
from PyKCCA. kernels import BowKernel, HistKernel
from PyKCCA import KCCA
from sklearn.metrics.pairwise import cosine_similarity

### GENERAL ###
def get_last_sentence(f_name):
    with open(f_name, 'rb') as fh:
        offs = -100
        while True:
            fh.seek(offs, 2)
            lines = fh.readlines()
            if len(lines)>1:
                last = lines[-1]
                break
            offs *= 2
        return last

def image_num(file_name):
    with open(file_name) as f:
        for i, l in enumerate(f):
            pass
    return (i + 1)/5

def get_splits(total, train_ratio, val_ratio):
    num_train = int(total*train_ratio)
    num_val = int(total*val_ratio)
    return num_train, num_val

### MICROSOFT ###
def get_split_ids_microsoft(domain, num_train, num_val, num_test):
    filename = os.path.join(domain, "SimpleSentences", "SimpleSentences1_10020.txt")
    last_sentence  = get_last_sentence(filename)
    total_image_num = int(last_sentence.split()[0]) + 1
    ids = [i for i in xrange(0, total_image_num)]
    shuffle(ids)
    return set(ids[:num_train]),\
           set(ids[num_train:num_train+num_val]),\
           set(ids[num_train+num_val:num_train+num_val+num_test])

def parse_microsoft_sentences(domain, train_ids, val_ids, test_ids):
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
                elif index in test_ids:
                    sentences = test_data
                else:
                    continue

                if index in sentences:
                    sentences[index] += " " + s[2]
                else:
                    sentences[index] = s[2]
    return train_data, val_data, test_data

def parse_microsoft_images(domain, train_ids, val_ids, test_ids):
    path = os.path.join(domain, "RenderedScenes")
    train_img = {}; val_img = {}; test_img = {}
    for f in os.listdir(path):
        # Ignore hidden filees
        if not f.startswith('.'):
            filepath = os.path.join(path, f)
            if os.path.isfile(filepath):
                nums = re.split(r'_|Scene|\.png', f)
                nums = filter(None, nums)
                nums = map(int, nums)
                index = nums[0]*10 + nums[1]
                if index in train_ids:
                    train_img[index] = filepath
                elif index in val_ids:
                    val_img[index] = filepath
                elif index in test_ids:
                    test_img[index] = filepath
                else:
                    continue

    return train_img, val_img, test_img

def parse_microsoft_dataset(domain, num_train, num_val, num_test):
    train_ids, val_ids, test_ids = get_split_ids_microsoft(domain, num_train, num_val, num_test)
    train_sen, val_sen, test_sen = parse_microsoft_sentences(domain, train_ids, val_ids, test_ids)
    train_img, val_img, test_img = parse_microsoft_images(domain, train_ids, val_ids, test_ids)
    return train_sen, train_img, val_sen, val_img, test_sen, test_img

### FLICKR ###
def get_split_ids_flickr30k(domain, num_train, num_val):
    sentences_path = os.path.join(domain, 'flickr30k', 'train_sentences.token')
    total_image_num = image_num(sentences_path)
    ids = [i for i in xrange(0, total_image_num)]
    shuffle(ids)
    return set(ids[:num_train]),\
           set(ids[num_train:num_train+num_val])

def parse_flickr30k_sentences(domain, train_index, val_index):
    filename = os.path.join(domain, "flickr30k", "train_sentences.token")
    test_filename = os.path.join(domain, "flickr30k", "test_sentences.token")
    train_data = {}; val_data = {}; test_data = {}; index = 0;
    prev_image_index = None

    # Process data for training and validation
    with open(filename) as f:
        f = f.readlines()
        for s in f:
            image_index = int(s.split('.', 1)[0])
            s = s.split('\t', 1)[1]
            s = s.strip()
            #Check if we are reading a sentence for the next image
            if prev_image_index and prev_image_index!=image_index:
                index +=1
            if index in train_index:
                sentences = train_data
            elif index in val_index:
                sentences = val_data
            else:
                prev_image_index = image_index
                continue

            if image_index in sentences:
                sentences[image_index] += " " + s
            else:
                sentences[image_index] = s
            prev_image_index = image_index

    # Process data for testing
    with open(test_filename) as f:
        f = f.readlines()
        for s in f:
            image_index = int(s.split('.', 1)[0])
            s = s.split('\t', 1)[1]
            s = s.strip()
            #Check if we are reading a sentence for the next image
            if prev_image_index and prev_image_index!=image_index:
                index +=1
            if image_index in test_data:
                test_data[image_index] += " " + s
            else:
                test_data[image_index] = s
            prev_image_index = image_index
    return train_data, val_data, test_data


def parse_flickr30k_images(domain, train_ids, val_ids, test_ids):
    path = os.path.join(domain, "flickr30k-images")
    train_img = {}; val_img = {}; test_img = {}
    for f in os.listdir(path):
        # Ignore hidden files
        if not f.startswith('.'):
            filepath = os.path.join(path, f)
            if os.path.isfile(filepath):
                index = int(f.split('.', 1)[0])
                if index in train_ids:
                    train_img[index] = filepath
                elif index in val_ids:
                    val_img[index] = filepath
                elif index in test_ids:
                    test_img[index] = filepath
                else:
                    continue
    return train_img, val_img, test_img

def parse_flickr30k_dataset(domain, train_num, val_num):
    train_ids, val_ids = get_split_ids_flickr30k(domain, train_num, val_num)
    train_sen, val_sen, test_sen = parse_flickr30k_sentences(domain, train_ids, val_ids)
    train_img, val_img, test_img = parse_flickr30k_images(domain, set(train_sen.keys()), set(val_sen.keys()), set(test_sen.keys()))
    return train_sen, train_img, val_sen, val_img, test_sen, test_img

def merge_two_dicts(x, y):
    '''Given two dicts, merge them into a new dict as a shallow copy.'''
    z = x.copy()
    z.update(y)
    return z

def find_files(dictionary):
    count_w = 0
    count_r = 0
    for value in dictionary.values():
        if not os.path.exists(value):
            count_w += 1
            print value
        else:
            count_r += 1
    print "Could not find", count_w, "files"
    print "Found", count_r, "files"

def write_to_file(scores, filename):
    with open(filename, 'w') as fw:
        for i in scores:
            fw.write(str(i))
            fw.write('\n')
        fw.close()

def re_index(dictionary_sen, dictionary_img):
    value_sen = [0.0]*len(dictionary_sen)
    value_img = [0.0]*len(dictionary_sen)
    index = 0
    for key in dictionary_sen:
        value_sen[index] = dictionary_sen[key]
        value_img[index] = dictionary_img[key]
        index += 1
    return value_sen, value_img

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
    value_sen_test = value_sen_test[:20]
    value_img_test = value_img_test[:20]
    if not opts.random:

        #sentences.train_lda(value_sen, 10, opts.out_sentence)
        sentences.train_bow(value_sen_train, opts.out_sentence, opts.out_file)
        images.trainSift(images.PathSet(value_img_train), 5, opts.out_image, opts.out_file)
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

        #cca = CCA(n_components=35)
        #cca.fit(train_sen_feat, train_img_feat)
        #cca = rcca.CCA(kernelcca=False, numCC=2, reg=0.)
        #cca.train([train_sen_feat, train_img_feat])

        #print "CCA done"
        #print cca

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
        print "test_sen_cca", test_sen_c.shape
        print "test_img_cca", test_img_c.shape
        #ev = cca.compute_ev([test_sen_feat, test_img_feat])
        #print ev

        print "Testing set after transformation"
        print "Sentences: ", test_sen_c.shape
        print "Images: ", test_img_c.shape

        l1_all = {}; l2_all = {}; cosine_all = {} # cosine is similarity measure
        pred_index_l1 = []; pred_index_l2 = []; pred_index_cosine = []
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
