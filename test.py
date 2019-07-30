# from questionAnswering.questionAnalysis import build_question_database
# PATH = 'data/inData/natural_questions/v1.0/train'
# x = build_question_database(PATH, n=2500, outPath='data/outData/paragraphFindingDf.sav')


# import questionAnswering.squadAnalysis

from bert_serving.client import BertClient
from scipy.spatial.distance import cosine
from vectorizers.docVecs import get_word_encodings

bc = BertClient(check_length=False)


while True:
    s = input("s: ")
    print(len(get_word_encodings(s)))
