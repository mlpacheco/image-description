import os
from random import shuffle, randint
import re
import numpy as np
import json
import xml.etree.ElementTree as ET

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

def centroid(a):
    if a:
        return[np.mean(np.array(a)[:,i]) for i in range(len(a[0]))]
    else:
        return np.array([2000,2000])

### MICROSOFT ###
def get_mirosoft_entity(png_id, categories):
    if png_id[0] == 's':
        return categories[png_id.split('.')[0]]
    else:
        return categories[png_id.split('_')[0]]

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
                    sentences[index][0] += " " + s[2]
                else:
                    sentences[index] = [s[2],0]
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

def parse_microsoft_entities(domain, train_ids, val_ids, test_ids):
    filename = os.path.join(domain, "Scenes_10020.txt")
    train_ent = {}; val_ent = {}; test_ent = {}
    categories = {}
    img_indexes = []
    with open('./model/general_categories.json') as f_json:
        categories = json.load(f_json)
    with open(filename) as f:
        f = f.readlines()
	count = 1
        for i in range(1002):
            for j in range(10):
                img_index, ent_total = map(int, f[count].split())
                index = img_index*10 + j
                img_indexes.append(index)
		count += 1
                selected_set = None
                if index in train_ids:
                    selected_set = train_ent
                elif index in val_ids:
                    selected_set = val_ent
                elif index in test_ids:
                    selected_set = test_ent
                else:
                    count += ent_total
                    continue
                c_points = [None]*5
		for k in range(ent_total):
                    ent_data = f[count].split()
                    c =  get_mirosoft_entity(ent_data[0], categories)
                    x = int(ent_data[3])
                    y = int(ent_data[4])
                    if c_points[c]:
                        c_points[c].append([x,y])
                    else:
                        c_points[c] = [[x,y]]
                    count += 1
                selected_set[index] = np.array([centroid(e) for e in c_points]).flatten()

    return train_ent, val_ent, test_ent


def parse_microsoft_dataset(domain, num_train, num_val, num_test):
    train_ids, val_ids, test_ids = get_split_ids_microsoft(domain, num_train, num_val, num_test)
    train_sen, val_sen, test_sen = parse_microsoft_sentences(domain, train_ids, val_ids, test_ids)
    train_img, val_img, test_img = parse_microsoft_images(domain, train_sen.keys(), val_sen.keys(), test_sen.keys())
    train_ent, val_ent, test_ent = parse_microsoft_entities(domain, train_sen.keys(), val_sen.keys(), test_sen.keys())
    return train_sen, train_img, train_ent, val_sen, val_img, val_ent, test_sen, test_img, test_ent

### FLICKR ###
def get_flickr_category(string):
    if string == 'people':
        return 0
    if string == 'clothing':
        return 1
    if string == 'animals':
        return 2
    if string == 'vehicles':
        return 3
    if string == 'other':
        return 4
    else:
        return -1
def get_coordinates(o):
    xmax = int(o.find('bndbox').find('xmax').text)
    xmin = int(o.find('bndbox').find('xmin').text)
    ymax = int(o.find('bndbox').find('ymax').text)
    ymin = int(o.find('bndbox').find('ymin').text)
    return (xmax + xmin)/2.0, (ymax + ymin)/2.0


def get_flickr_position(string, xml):
    for o in xml.findall('object'):
        if string == o.find('name').text:
            if o.find('bndbox')!=None:
                return get_coordinates(o)
    for o in xml.findall('object'):
        if string in [e.text for e in o.findall('name')]:
            if o.find('bndbox')!=None:
                return get_coordinates(o)
            else:
                return None, None

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
                sentences[image_index][0] += " " + s
            else:
                sentences[image_index] = [s, 1]
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
                test_data[image_index][0] += " " + s
            else:
                test_data[image_index] = [s, 1]
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

def parse_flickr30k_entities(domain, train_ids, val_ids, test_ids):
    path_sentence = os.path.join(domain, "Flickr30kEntities/Sentences")
    path_data = os.path.join(domain, "Flickr30kEntities/Annotations")
    train_ent = {}; val_ent = {}; test_ent = {}
    for f in os.listdir(path_sentence):
        # Ignore hidden files
        if not f.startswith('.'):
            filepath = os.path.join(path_sentence, f)
            if os.path.isfile(filepath):
                index = int(f.split('.', 1)[0])
                if index in train_ids:
                    selected_set = train_ent
                elif index in val_ids:
                    selected_set = val_ent
                elif index in test_ids:
                    selected_set = test_ent
                else:
                    continue
                ent_sentences = []
                # Parse sentences with entities
                with open(filepath) as ent_file:
                    for s in ent_file.readlines():
                        ent_sentences += re.findall('/EN#(\d+)/([A-Za-z]+)',s)
                # Parse xml with entities positions
                xml_filepath = os.path.join(path_data, f.replace('txt','xml'))
                xml = ET.parse(xml_filepath)
                c_points = [None]*5
                for elem in ent_sentences:
                    c =  get_flickr_category(elem[1])
                    if c < 0:
                        continue
                    x, y = get_flickr_position(elem[0],xml)
                    if x==None and y==None:
                        continue
                    if c_points[c]:
                        c_points[c].append((x,y))
                    else:
                        c_points[c] = [(x,y)]
                selected_set[index] = np.array([centroid(e) for e in c_points]).flatten()

    return train_ent, val_ent, test_ent

def parse_flickr30k_dataset(domain, train_num, val_num):
    train_ids, val_ids = get_split_ids_flickr30k(domain, train_num, val_num)
    train_sen, val_sen, test_sen = parse_flickr30k_sentences(domain, train_ids, val_ids)
    train_img, val_img, test_img = parse_flickr30k_images(domain, set(train_sen.keys()), set(val_sen.keys()), set(test_sen.keys()))
    train_ent, val_ent, test_ent = parse_flickr30k_entities(domain, set(train_sen.keys()), set(val_sen.keys()), set(test_sen.keys()))
    return train_sen, train_img, train_ent, val_sen, val_img, val_ent, test_sen, test_img, test_ent

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

def re_index(dictionary_sen, dictionary_img, dictionary_ent):
    value_sen = [0.0]*len(dictionary_sen)
    value_img = [0.0]*len(dictionary_sen)
    value_ent = [0.0]*len(dictionary_sen)
    index = 0
    for key in dictionary_sen:
        value_sen[index] = dictionary_sen[key]
        value_img[index] = dictionary_img[key]
        value_ent[index] = dictionary_ent[key]
        index += 1
    return value_sen, value_img, value_ent
