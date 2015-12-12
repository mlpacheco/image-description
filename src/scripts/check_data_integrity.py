import optparse
import os

def check_flickr(domain):
    prepname1 = os.path.join(domain, "train_sentences_preproc.token")
    prepname2 = os.path.join(domain, "test_sentences_preproc.token")
    posname1 = os.path.join(domain, "train_sentences_pos.token")
    posname2 = os.path.join(domain, "test_sentences_pos.token")

    check(prepname1, posname1)
    check(prepname2, posname2)

def check_microsoft(domain):
    prepname1 = os.path.join(domain, "SimpleSentences", "newSimpleSentences1_10020_preproc.txt")
    prepname2 = os.path.join(domain, "SimpleSentences", "newSimpleSentences2_10020_preproc.txt")
    posname1 = os.path.join(domain, "SimpleSentences", "newSimpleSentences1_10020_pos.txt")
    posname2 = os.path.join(domain, "SimpleSentences", "newSimpleSentences2_10020_pos.txt")

    check(prepname1, posname1)
    check(prepname2, posname2)

def check(prepfile, posfile):
    a = open(prepfile); b = open(posfile)
    count = 0
    for x,y in zip(a.readlines(), b.readlines()):
        if len(x.split()) != len(y.split()):
            print x.split()[0]
            count += 1
    print "Encountered", count, "disparities"

parser = optparse.OptionParser()
parser.add_option('-p', '--path', help='path to the sentences files', type='string')
parser.add_option('-d', '--domain', help='0: microsoft, 1: flickr', type='int')
(opts, args) = parser.parse_args()

if opts.domain == 0:
    check_microsoft(opts.path)
elif opts.domain == 1:
    check_flickr(opts.path)
else:
    parser.print_help()


