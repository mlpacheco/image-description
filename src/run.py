import optparse
import os
from random import shuffle
import re
import numpy as np

# features
import sentences
import images

# algorithms
from sklearn.cross_decomposition import CCA
from sklearn.metrics.pairwise import pairwise_distances

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
def get_split_ids_microsoft(domain, train_ratio, val_ratio):
    filename = os.path.join(domain, "SimpleSentences", "SimpleSentences1_10020.txt")
    last_sentence  = get_last_sentence(filename)
    total_image_num = int(last_sentence.split()[0]) + 1
    num_train, num_val = get_splits(total_image_num, train_ratio, val_ratio)
    ids = [i for i in xrange(0, total_image_num)]
    shuffle(ids)
    return set(ids[:num_train]),\
           set(ids[num_train:num_train+num_val])

def parse_microsoft_sentences(domain, train_ids, val_ids):
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

def parse_microsoft_images(domain, train_ids, val_ids):
    path = os.path.join(domain, "RenderedScenes")
    train_img = {}; val_img = {}; test_img = {}
    for f in os.listdir(path):
        # Ignore hidden files
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
                else:
                    test_img[index] = filepath
    return train_img, val_img, test_img

def parse_microsoft_dataset(domain, train_ratio, val_ratio):
    train_ids, val_ids = get_split_ids_microsoft(domain, train_ratio, val_ratio)
    train_sen, val_sen, test_sen = parse_microsoft_sentences(domain, train_ids, val_ids)
    train_img, val_img, test_img = parse_microsoft_images(domain, train_ids, val_ids)
    return train_sen, train_img, val_sen, val_img, test_sen, test_img

### FLICKR ###
def get_split_ids_flickr30k(domain, train_ratio, val_ratio):
    sentences_path = os.path.join(domain, 'flickr30k', 'results_20130124.token')
    total_image_num = image_num(sentences_path)
    num_train, num_val = get_splits(total_image_num, train_ratio, val_ratio)
    ids = [i for i in xrange(0, total_image_num)]
    shuffle(ids)
    return set(ids[:num_train]),\
           set(ids[num_train:num_train+num_val])

def parse_flickr30k_sentences(domain, train_index, val_index):
    filename = os.path.join(domain, "flickr30k", "results_20130124.token")
    train_data = {}; val_data = {}; test_data = {}; index = 0;
    prev_image_index = None
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
                sentences = test_data

            if image_index in sentences:
                sentences[image_index] += " " + s
            else:
                sentences[image_index] = s
            prev_image_index = image_index
    return train_data, val_data, test_data


def parse_flickr30k_images(domain, train_ids, val_ids):
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
                else:
                    test_img[index] = filepath
    return train_img, val_img, test_img

def parse_flickr30k_dataset(domain, train_ratio, val_ratio):
    train_ids, val_ids = get_split_ids_flickr30k(domain, train_ratio, val_ratio)
    train_sen, val_sen, test_sen = parse_flickr30k_sentences(domain, train_ids, val_ids)
    train_img, val_img, test_img = parse_flickr30k_images(domain, set(train_sen.keys()), set(val_sen.keys()))
    return train_sen, train_img, val_sen, val_img, test_sen, test_img

def merge_two_dicts(x, y):
    '''Given two dicts, merge them into a new dict as a shallow copy.'''
    z = x.copy()
    z.update(y)
    return z

### MAIN ###
def main():
    parser = optparse.OptionParser()
    parser.add_option('-s', '--source', help='path to source domain dir',
                      dest='source', type='string')
    parser.add_option('-t', '--target', help='path to target domain dir',
                      dest='target', type='string')
    parser.add_option('--so', help='output path for sentence files',
                      dest='out_sentence', type='string')
    parser.add_option('--io', help='output path for image files',
                      dest='out_image', type='string')
    parser.add_option('--rts', help='ratio of examples for training',
                      dest='train_ratio_source', type='float', default=1.0)
    parser.add_option('--rtg', help='ratio of examples for training',
                      dest='train_ratio_target', type='float', default=0.3)
    #parser.add_option('--rvs', help='ratio of examples for validation', ## pasarle 0.5
    #                  dest='val_ratio', type='int', default=0.0)
    (opts, args) = parser.parse_args()

    mandatories = ['source', 'target', 'out_image', 'out_sentence']
    for m in mandatories:
        if not opts.__dict__[m]:
            print "mandatory option is missing\n"
            parser.print_help()
            exit(-1)

    print "PARSING################"
    f_train_sen, f_train_img, f_val_sen, f_val_img, f_test_sen, f_test_img = parse_flickr30k_dataset(opts.target, opts.train_ratio_target, 0.0)
    m_train_sen, m_train_img, m_val_sen, m_val_img, m_test_sen, m_test_img = parse_microsoft_dataset(opts.source, opts.train_ratio_source, 0.0)


    print "Training with -> ", "Microsoft: ", len(m_train_sen), "Flickr: ", len(f_train_sen)

    train_sen = merge_two_dicts(f_train_sen, m_train_sen)
    train_img = merge_two_dicts(f_train_img, m_train_img)
    test_sen = f_test_sen
    test_img = f_test_img
    print "Parsing complete"

    #print "Training###############"
    #print train_sen
    #print train_img
    #print "Validation################"
    #print val_sen
    #print val_img
    #print "testing################"
    #print test_sen
    #print test_img

    #sentences.train_lda(train_sen, 10, opts.out_sentence)
    sentences.train_bow(train_sen, opts.out_sentence)
    images.trainSift(images.PathSet(train_img.values()), 256, opts.out_image)
    print "Training features complete"

    #train_sen_feat = sentences.extract_lda(train_sen, 10, opts.out_sentence)
    train_sen_feat = sentences.extract_bow(train_sen, opts.out_sentence).toarray()
    train_img_feat = images.FeaturesMatrix()
    images.extractFeats(opts.out_image, images.PathSet(train_img.values()), train_img_feat)
    train_img_feat = np.asarray(train_img_feat)
    print "Extraction of feats complete"
    print "Sentences: ", train_sen_feat.shape
    print "Images: ", train_img_feat.shape

    cca = CCA(n_components=9)
    cca.fit(train_sen_feat, train_img_feat)

    print "CCA done"
    print cca

    #sen_train_c, img_train_c = cca.transform(train_sen_feat, train_img_feat)


    test_sen_feat = sentences.extract_bow(test_sen, opts.out_sentence).toarray()
    test_img_feat = images.FeaturesMatrix()
    images.extractFeats(opts.out_image, images.PathSet(test_img.values()), test_img_feat)

    test_sen_c, test_img_c = cca.transform(test_sen_feat, test_img_feat)
    print "Testing set after transformation"
    print "Sentences: ", test_sen_c.shape
    print "Images: ", test_img_c.shape

    l1_all = {}; l2_all = {}; cosine_all = {} # cosine is similarity measure

    for i in xrange(0, len(test_sen_c)):
        X = test_sen_c[i]
        l1 = []; l2 = []; cosine = [];
        for j in xrange(0, len(test_sen_c)):
            Y = test_img_c[j]
            index = test_sen.keys()[j]
            l1.append((pairwise_distances(X, Y, metric='l1')[0][0], j))
            l2.append((pairwise_distances(X, Y, metric='l2')[0][0], j))
            cosine.append((pairwise_distances(X, Y, metric='cosine')[0][0], j))

        l1.sort()
        l1 = l1[:5]
        l2.sort()
        l2 = l2[:5]
        cosine.sort(reverse=1)
        cosine = cosine[:5]

        l1_all[i] = [index for (metric,index) in l1]
        l2_all[i] = [index for (metric,index) in l2]
        cosine_all[i] = [index for (metric,index) in cosine]

    for r in [1,5,10]:
        hits_l1 = 0.0; hits_l2 = 0.0; hits_cosine = 0.0
        for i in xrange(0, len(test_sen_c)):

            index = test_sen.keys()[i]
            if index in l1_all[i][:r]:
                hits_l1+=1
            if index in l2_all[i][:r]:
                hits_l2+=1
            if index in cosine_all[i][:r]:
                hits_cosine+=1

        print "R@{0} w/ L1: ".format(r), hits_l1/len(test_sen_c)
        print "R@{0} w/ L2: ".format(r), hits_l2/len(test_sen_c)
        print "R@{0} w/ Cosine: ".format(r), hits_cosine/len(test_sen_c)




if __name__ == "__main__":
    main()
