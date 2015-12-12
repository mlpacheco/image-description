import re

import numpy
from sklearn.metrics.pairwise import cosine_similarity, polynomial_kernel

import images


class DomainAdaptation(object):

    def __init__(self, domain_adapt):
        self.domain_adapt = domain_adapt

    def same_image_domain(self, path1, path2):
        if (re.match(r'.*RenderedScenes.*', path1) and
            re.match(r'.*RenderedScenes.*', path2)):
            return True
        elif (re.match(r'.*flickr30k-images.*', path1) and
              re.match(r'.*flickr30k-images.*', path2)):
            return True
        else:
            return False

    def same_sentence_domain(self, domain1, domain2):
        return domain1 == domain2


class BowKernel(DomainAdaptation):

    def __init__(self, domain_adapt=False):
        super(BowKernel, self).__init__(domain_adapt)

    def __call__(self, X1, X2):
        """ receives as X1 and X2 a list of tuples (tfidf-vector, domain_index) """
        m = []
        for (x1,domain1) in X1:
            ret = []
            for (x2,domain2) in X2:
                sim = cosine_similarity(x1.reshape(1, -1), x2.reshape(1, -1))[0][0]
                if self.domain_adapt and self.same_sentence_domain(domain1, domain2):
                    ret.append(2*sim)
                else:
                    ret.append(sim)
            m.append(ret)
        m = numpy.asarray(m)
        print m.shape
        return m

class HistKernel(DomainAdaptation):

    def __init__(self, trained_path, out_file, domain_adapt=False):
        super(HistKernel, self).__init__(domain_adapt)
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
        for x1s,x1c,x1 in zip(X1_sift,X1_cielab, X1):
            ret = []
            for x2s,x2c,x2 in zip(X2_sift, X2_cielab, X2):
                inter_sift = images.intersectionScore(x1s, x2s)
                inter_cielab = images.intersectionScore(x1c, x2c)
                sim = (inter_sift+inter_cielab)/2.0
                if self.domain_adapt and self.same_image_domain(x1, x2):
                    ret.append(2*sim)
                else:
                    ret.append(sim)
            m.append(ret)
        m = numpy.asarray(m)
        print m.shape
        return m

def PyramidKernel(DomainAdaptation):

    def __init__(self, domain_adapt):
        super(PyramidKernel, self).__init__(domain_adapt)

    def __call__(self, X1, X2):
        pass


class EntitiesKernel(DomainAdaptation):

    def __init__(self, domain_adapt=False):
        super(EntitiesKernel, self).__init__(domain_adapt)

    def __call__(self, X1, X2):
        return polynomial_kernel(X1, X2, degree=2)
