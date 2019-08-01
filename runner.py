import numpy as np
import pandas as pd
from os import listdir
from nltk.tokenize import word_tokenize
from scipy.spatial.distance import cosine
from scipy.special import softmax
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.regularizers import l2
from bert_serving.client import BertClient


bc = BertClient()

context = """ Google was founded in 1998 by Larry Page and Sergey Brin while they were Ph.D. students at Stanford University in California. Together they own about 14 percent of its shares and control 56 percent of the stockholder voting power through supervoting stock. They incorporated Google as a privately held company on September 4, 1998. An initial public offering (IPO) took place on August 19, 2004, and Google moved to its headquarters in Mountain View, California, nicknamed the Googleplex. In August 2015, Google announced plans to reorganize its various interests as a conglomerate called Alphabet Inc. Google is Alphabet's leading subsidiary and will continue to be the umbrella company for Alphabet's Internet interests. Sundar Pichai was appointed CEO of Google, replacing Larry Page who became the CEO of Alphabet. """

contextTokens = word_tokenize(context)






# print(contextTokens[107:109])
#
# contextVec = bc.encode([contextTokens], is_tokenized=True)[0][1:-2]
#
# while True:
#     question = input('Q: ')
#     questionTokens = word_tokenize(question)
#     questionVec = bc.encode([questionTokens], is_tokenized=True)[0]
#     startVec = questionVec[0]
#     endVec = questionVec[-1]
#
#     print(questionVec)
#
#     calc_start_score = lambda wordVec : np.dot(wordVec, startVec)
#     calc_end_score = lambda wordVec : np.dot(wordVec, endVec)
#
#     # iterate over contextVec and get dot product of
#     startScores = softmax([calc_start_score(wordVec) for wordVec in contextVec])
#     startScores = [elt for elt in startScores]
#
#     startLoc = startScores.index(max(startScores))
#
#     endScores = softmax([calc_start_score(wordVec) for wordVec in contextVec[(startLoc+1): (startLoc + 10)]])
#
#
#     endScores = [elt for elt in endScores]
#
#
#     # startLoc = startScores.index(max(startScores))
#     endLoc = endScores.index(max(endScores)) + startLoc + 1
#
#     print(startLoc, endLoc)
#
#     print(contextTokens[startLoc:endLoc])
