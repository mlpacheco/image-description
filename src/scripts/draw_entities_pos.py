import optparse
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np

def parse_args():
    parser = optparse.OptionParser()
    parser.add_option('-i', '--images', help='path to the images directory',\
                      dest='img_path', type='string')
    parser.add_option('-s', '--source', help='path to the schene description file',\
                      dest='sentence_path', type='string')
    (opts, args) = parser.parse_args()
    return opts

def parse_sentences(file_path, img_path):
    positionns = []
    filepaths = []
    with open(file_path) as f:
        f = f.readlines()
        total_img = int(f[0])
	count = 1
        for i in range(total_img):
            for j in range(10):
                img_index, ent_total = map(int, f[count].split())
		count += 1
		for k in range(ent_total):
                    ent_data = f[count].split()
                    x = int(ent_data[3])
                    y = int(ent_data[4])
                    count += 1
                    paint_pos(img_path, img_index, j, x, y)

    return positionns, filepaths

def paint_pos(img_path, img_index, scene_index, x, y):
    filename = join(img_path, 'Scene'+str(img_index)+"_"+str(scene_index)+".png")
    print filename
    img = cv2.imread(filename)
    cv2.circle(img, (x,y), 10, (255,0,0), 4)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass

def main():
    opts = parse_args()
    parse_sentences(opts.sentence_path, opts.img_path)

if __name__ == "__main__":
    main()
