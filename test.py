# from questionAnswering.questionAnalysis import build_question_database
# PATH = 'data/inData/natural_questions/v1.0/train'
# x = build_question_database(PATH, n=2500, outPath='data/outData/paragraphFindingDf.sav')


# import questionAnswering.squadAnalysis

import re
import numpy as np
from functools import reduce
from bert_serving.client import BertClient
from scipy.spatial.distance import cosine

bc = BertClient(check_length=False)

def get_word_encodings(text):
    """
    Returns ordered list of word vec embeddings in context of their sentence.
    Can handle text of any length, but sentences over 25 words long will
    be split in two.
    Does not include CLS or SEP tokens but does include punctuation.
    """
    sentences = re.split(r'(?<=[\.\!\?])[^a-zA-Z0-9]+', text)
    sentenceVecs = bc.encode(sentences)
    textVecs =
    print(textVecs)
    # for sentence in sentences:
    #     sentenceWords = sentence.split()
    #     print(sentenceWords)
    #     for i, wordVec in enumerate(sentenceVecs):
    #         word = sentenceWords[i]


while True:
    s = input("s: ")
    get_word_encodings(s)
    # sVec = bc.encode([s])
    # sWords = ['CLS'] + s.split() + ['SEP']
    # for i, wordVec in enumerate(sVec[0]):
    #     if not wordVec[0]==0:
    #         print(sWords[i], wordVec)
    #         print(np.sum(wordVec))
