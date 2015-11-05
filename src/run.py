import optparse
import os
from random import shuffle
import re

# features
import sentences
import images

# algorithms

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

def get_splits(total, train_ratio, val_ratio):
    num_train = int(total*train_ratio)
    num_val = int(total*val_ratio)
    return num_train, num_val

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


def parse_flickr30k_sentences(domain):
    pass


def parse_flickr30k_images(domain):
    pass


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
    parser.add_option('--rt', help='ratio of examples for training',
                      dest='train_ratio', type='int', default=0.7)
    parser.add_option('--rv', help='ratio of examples for validation',
                      dest='val_ratio', type='int', default=0.0)
    (opts, args) = parser.parse_args()

    mandatories = ['source', 'target', 'out_image', 'out_sentence']
    for m in mandatories:
        if not opts.__dict__[m]:
            print "mandatory option is missing\n"
            parser.print_help()
            exit(-1)

    train_sen, train_img, val_sen, val_img, test_sen, test_img = parse_microsoft_dataset(opts.source, opts.train_ratio, opts.val_ratio)

    #sentences.train_lda(train_sen, 10, opts.out_sentence)
    sentences.train_bow(train_sen, opts.out_sentence)
    images.trainSift(images.PathSet(train_img.values()), 256, opts.out_image)


if __name__ == "__main__":
    main()
