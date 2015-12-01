import optparse
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np

def parse_args():
    parser = optparse.OptionParser()
    parser.add_option('-i', '--images', help='path to the images directory',\
                      dest='img_path', type='string')
    parser.add_option('-s', '--source', help='path to the captions directory',\
                      dest='sentence_path', type='string')
    (opts, args) = parser.parse_args()
    return opts

def grass_empty(img, range_init, colors):
    height, width, channels = img.shape
    for i in range(range_init, height):
        for j in range(width):
            pixel = img[i][j]
            if not np.array_equal(pixel, colors):
                return False
    return True

def get_null_images(img_path, range_init, colors):
    null_files = []
    img_files = [f for f in listdir(img_path) if isfile(join(img_path, f)) and not f.startswith('.')]
    for f in img_files:
        img = cv2.imread(join(img_path, f))
        if grass_empty(img, range_init, colors):
            null_files.append(f)
    return null_files

def generate_deleted_sentences(domain, null_files):
    filename1 = "SimpleSentences1_10020.txt"
    filename2 = "SimpleSentences2_10020.txt"
    null_ids = [map(int, s.split('.')[0][5:].split('_')) for s in null_files]
    null_ids = [i[0]*10 + i[1] for i in null_ids]
    deleted_count = 0
    for filename in [filename1, filename2]:
        new_filename = join(domain, "new" + filename)
        filename = join(domain, filename)
        with open(filename) as f, open(new_filename, 'w+') as fn:
            f = f.readlines()
            for s in f:
                sen = s.strip().split("\t", 2)
                index = int(sen[0])
                if index not in null_ids:
                    fn.write(s)
                else:
                    deleted_count += 1
    return deleted_count

def main():
    opts = parse_args()
    null_files = get_null_images(opts.img_path, 191, [84, 163,  89])
    deleted_count = generate_deleted_sentences(opts.sentence_path, null_files)
    print 'Deleted:', deleted_count, 'sentences'

if __name__ == "__main__":
    main()
