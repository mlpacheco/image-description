import os
import random
import optparse

def split_flickr_images(filename):
    test_filename = os.path.join(os.path.dirname(filename), 'test_sentences.token')
    train_filename = os.path.join(os.path.dirname(filename), 'train_sentences.token')
    flickr_img_ids = []
    test_f = open(test_filename,'w')
    train_f = open(train_filename,'w')
    with open(filename) as f:
        # Get all image keys and put them in a set
        f = f.readlines()
        for s in f:
            image_index = int(s.split('.', 1)[0])
            flickr_img_ids.append(image_index)
        test_img_keys = set(random.sample(set(flickr_img_ids), 1000))

    with open(filename) as f:
        f = f.readlines()
        for s in f:
            image_index = int(s.split('.', 1)[0])
            if image_index in test_img_keys:
                test_f.write(s)
            else:
                train_f.write(s)
        test_f.close()
        train_f.close()

def main():
    parser = optparse.OptionParser()
    parser.add_option('-f', '--flickr', help='path to the Flickr sentences file',
                      dest='flickr_path', type='string')
    parser.add_option('-m', '--microsoft', help='path to the Microsoft sentences file',
                      dest='microsoft_path', type='string')
    (opts, args) = parser.parse_args()

    # Split the flickr images
    split_flickr_images(opts.flickr_path)

if __name__ == "__main__":
    main()
