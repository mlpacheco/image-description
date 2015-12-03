import os
from random import shuffle, randint
import re
import numpy as np

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
    filename = os.path.join(domain, "SimpleSentences", "newSimpleSentences1_10020.txt")
    last_sentence  = get_last_sentence(filename)
    total_image_num = int(last_sentence.split()[0]) + 1
    ids = [i for i in xrange(0, total_image_num)]
    shuffle(ids)
    return set(ids[:num_train]),\
           set(ids[num_train:num_train+num_val]),\
           set(ids[num_train+num_val:num_train+num_val+num_test])

def parse_microsoft_sentences(domain, train_ids, val_ids, test_ids):
    filename1 = os.path.join(domain, "SimpleSentences", "newSimpleSentences1_10020.txt")
    filename2 = os.path.join(domain, "SimpleSentences", "newSimpleSentences2_10020.txt")
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

def re_index(dictionary_sen, dictionary_img):
    value_sen = [0.0]*len(dictionary_sen)
    value_img = [0.0]*len(dictionary_sen)
    index = 0
    for key in dictionary_sen:
        value_sen[index] = dictionary_sen[key]
        value_img[index] = dictionary_img[key]
        index += 1
    return value_sen, value_img

