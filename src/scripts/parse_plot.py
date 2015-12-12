import optparse
import matplotlib.pyplot as plt
import numpy as np

def parse_file(filepath):
    with open(filepath) as f:
        ret = f.read().strip().split('\n')
        ret = map(int, ret)
    f.close()
    return ret

def percentages(rank):
    ret = []
    for i in xrange(1000):
        rate = sum(j <= i for j in rank)
        ret.append((rate/1000.0)*100)
    return ret

def print_stats(rank):
    rank = np.asarray(rank)
    print "Min: ", np.min(rank)
    print "Avg: ", np.mean(rank)
    print "Med: ", np.median(rank)
    print "Std: ", np.std(rank)
    #print "Var: ", np.var(rank)
    print "Max: ", np.max(rank)
    return rank

parser = optparse.OptionParser()
parser.add_option('-f', '--files', help='input files (multiple)', dest='in_files', action='append')
parser.add_option('-o', '--out', help='output file', dest='out_file', type='string')
(opts, args) = parser.parse_args()

rank = parse_file(opts.in_files[0])
print_stats(rank)

results = percentages(parse_file(opts.in_files[0]))
print results[10], results[30], results[50], results[70], results[100], results[250], results[500]

#plt.plot(results)
#plt.show()
