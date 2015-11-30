import numpy
from sklearn.metrics.pairwise import cosine_similarity

import images

class CrossModalKernel(object):
    """ parent class with static variable to handle bad indexes """
    bad_image_indexes_1 = None
    bad_image_indexes_2 = None

class BowKernel(CrossModalKernel):

    def __init__(self):
        pass

    def __call__(self, X1, X2):
        """ receives X1 and X2 tf-idf matrix """
        X1 = numpy.delete(X1, tuple(CrossModalKernel.bad_image_indexes_1), axis=0)
        X2 = numpy.delete(X2, tuple(CrossModalKernel.bad_image_indexes_2), axis=0)

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

class HistKernel(CrossModalKernel):

    def __init__(self, trained_path, out_file):
        self.trained_path = trained_path
        self.out_file = out_file

    def __call__(self, X1, X2):
        """ receives X1 and X2 sets of image path """
        """ sets static variable to empty for sync with text kernels """
        """ extracts needed features for kernel computations """
        CrossModalKernel.bad_image_indexes_1 = images.BadIndexes()
        CrossModalKernel.bad_image_indexes_2 = images.BadIndexes()

        X1_sift = images.FeaturesMatrix(); X2_sift = images.FeaturesMatrix()

        images.extractFeats(self.trained_path, images.PathSet(X1), X1_sift,
                            CrossModalKernel.bad_image_indexes_1, self.out_file)
        if X1[0] != X2[0]:
            images.extractFeats(self.trained_path, images.PathSet(X2), X2_sift,
                                CrossModalKernel.bad_image_indexes_2, self.out_file)
        else:
            X2_sift = X1_sift

        m = []
        for x1 in X1_sift:
            ret = []
            for x2 in X2_sift:
                inter = images.intersectionScore(x1, x2)
                ret.append(inter)
            m.append(ret)
        m = numpy.asarray(m)
        print m.shape
        return m

def PyramidKernel(CrossModalKernel):

    def __init__(self):
        pass

    def __call__(self, X1, X2):
        pass
