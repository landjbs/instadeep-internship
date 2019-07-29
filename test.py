# from questionAnswering.questionAnalysis import build_question_database
# PATH = 'data/inData/natural_questions/v1.0/train'
# x = build_question_database(PATH, n=2500, outPath='data/outData/paragraphFindingDf.sav')

import re
from scipy.spatial.distance import euclidean

from vectorizers.docVecs import vectorize_doc

text = """I recently completed a course on NLP through Deep Learning (CS224N) at Stanford and loved the experience.
        Learnt a whole bunch of new things. For my final project I worked on a question answering model built on
        Stanford Question Answering Dataset (SQuAD). In this blog, I want to cover the main building blocks of a
        question answering model. """

print(text)
