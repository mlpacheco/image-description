import numpy
from sklearn.metrics.pairwise import cosine_similarity

import images


class BowKernel(object):

    def __init__(self):
        pass

    def __call__(self, X1, X2):
        """ receives X1 and X2 tf-idf matrix """
        if X1.shape[1] != X2.shape[1]:
            raise ValueError("Invalid matrix dimensions: " + str(X1.shape) + " " + str(X2.shape))

        m = []
        for x1 in X1:
            ret = []
            for x2 in X2:
                sim = cosine_similarity(x1.reshape(1, -1), x2.reshape(1, -1))[0][0]
                ret.append(sim)
            m.append(ret)
        m = numpy.asarray(m)
        print m.shape
        return m

class HistKernel(object):

    def __init__(self, trained_path, out_file):
        self.trained_path = trained_path
        self.out_file = out_file

    def __call__(self, X1, X2):
        """ receives X1 and X2 sets of image path """
        """ sets static variable to empty for sync with text kernels """
        """ extracts needed features for kernel computations """

        X1_sift = images.FeaturesMatrix(); X2_sift = images.FeaturesMatrix()
        X1_cielab = images.FeaturesMatrix(); X2_cielab = images.FeaturesMatrix()

        images.extractFeats(self.trained_path, images.PathSet(X1),
                            X1_sift, X1_cielab, self.out_file)
        if X1[0] != X2[0]:
            images.extractFeats(self.trained_path, images.PathSet(X2),
                                X2_sift, X2_cielab, self.out_file)
        else:
            X2_sift = X1_sift
            X2_cielab = X1_cielab

        m = []
        for x1s,x1c in zip(X1_sift,X1_cielab):
            ret = []
            for x2s,x2c in zip(X2_sift, X2_cielab):
                inter_sift = images.intersectionScore(x1s, x2s)
                inter_cielab = images.intersectionScore(x1c, x2c)
                ret.append((inter_sift+inter_cielab)/2.0)
            m.append(ret)
        m = numpy.asarray(m)
        print m.shape
        return m

def PyramidKernel(object):

    def __init__(self):
        pass

    def __call__(self, X1, X2):
        pass
