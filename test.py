# from questionAnswering.questionAnalysis import build_question_database
# PATH = 'data/inData/natural_questions/v1.0/train'
# x = build_question_database(PATH, n=2500, outPath='data/outData/paragraphFindingDf.sav')


# import questionAnswering.squadAnalysis

import re
import numpy as np
import matplotlib.pyplot as plt
from bert_serving.client import BertClient
from scipy.spatial.distance import cosine
bc = BertClient(check_length=False)

def get_word_encodings(text):
    """
    Returns ordered list of tuples mapping words to vec embedding in context of
    their sentence. Can handle text of any length, but sentences over 23 words
    long will have their ends ignored.
    Does not include CLS or SEP tokens but does include punctuation.
    """
    sentences = re.split(r'(?<=[\.\!\?])[^a-zA-Z0-9]+', text)
    sentenceVecs = bc.encode(sentences)
    print(len(sentenceVecs))


while True:
    s = input("s: ")
    get_word_encodings(s)
    # sVec = bc.encode([s])
    # sWords = ['CLS'] + s.split() + ['SEP']
    # for i, wordVec in enumerate(sVec[0]):
    #     if not wordVec[0]==0:
    #         print(sWords[i], wordVec)
    #         print(np.sum(wordVec))
