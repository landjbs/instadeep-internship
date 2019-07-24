import pickle
import numpy as np
import pandas as pd
from functools import reduce
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.stats import kurtosis

from utils.objectSaver import save, load
import vectorizers.docVecs as docVecs
from vectorizers.datasetVectorizer import vec_to_dict


class ProjectableCluster():
    """ Cluster """
    def __init__(self, wordSet=None, inPath=None, outPath=None):
        if wordSet:
            self.data = {word:docVecs.vectorize_doc(word)
                        for word in wordSet}
        elif inPath:
            self.data = load(inPath)
        else:
            raise ValueError('Valid load data must be given.')
        if outPath:
            save(self.data, outPath)
        print(f'{len(self.data)} words vectorized and loaded.')


    def find_nearest_simple(self, newVec, n=5):
        """ Finds nearest n words to newVec in pure euclidean space """
        wordVecs = self.data
        # find distances between each word and new word
        calc_dist = lambda word : euclidean(newVec, wordVecs[word])
        distList = [(calc_dist(word), word) for word in wordVecs]
        # sort the distList and take top neighbors
        distList.sort()
        return distList[:n]


    def find_nearest_via_projection(self, newVec, n=5, clusterSize=10, disp=True):
        """
        Finds nearest n words via pseudoprojection build from nearest clustSize
        words
        """
        # get list of top clustSize words and their vecs
        clusterList = [wordTuple[1] for wordTuple
                        in self.find_nearest_simple(newVec, n=clusterSize)]
        wordVecs = self.data
        # build dataframe of values in each dimension
        dimDf = pd.DataFrame([vec_to_dict(wordVecs[word])
                                    for word in clusterList])

        score_dimension = lambda dim : (1/(np.var(dimDf[dim])))

        # get the score of each dimension within the cluster
        dimensionScores = list(map(score_dimension, dimDf.columns))
        print([i for i, score in enumerate(dimensionScores) if score>1000])
        # recaluclate weighted euclidean distances to newVec
        calc_weighted_dist = lambda word : euclidean(newVec,
                                                    wordVecs[word],
                                                    w=dimensionScores)
        weightedDistList = [(calc_weighted_dist(word), word)
                            for word in self.data]
        weightedDistList.sort()
        return weightedDistList[:n]

    def find_nearest_via_centroid(self, newVec, n=5, clusterSize=10):
        # get list of top clustSize words and their vecs
        clusterList = [wordTuple[1] for wordTuple
                        in self.find_nearest_simple(newVec, n=clusterSize)]
        wordVecs = self.data
        # build dataframe of values in each dimension
        dimDf = pd.DataFrame([vec_to_dict(wordVecs[word])
                                    for word in clusterList])
        meanVec = dimDf.describe().loc['mean']
        scoreWord = lambda word : euclidean(meanVec, wordVecs[word])
        meanDists = [(scoreWord(word), word) for word in clusterList]
        print(euclidean(newVec, meanVec))
        meanDists.sort()
        return meanDists[:n]

    def find_nearest_mix(self, newVec, n=5, clusterSize=10):
        projectionResults = self.find_nearest_via_projection(newVec, n=5)
        centroidResults = self.find_nearest_via_centroid(newVec, n=5)
        print(projectionResults, centroidResults)
