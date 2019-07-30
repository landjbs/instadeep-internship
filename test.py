# from questionAnswering.questionAnalysis import build_question_database
# PATH = 'data/inData/natural_questions/v1.0/train'
# x = build_question_database(PATH, n=2500, outPath='data/outData/paragraphFindingDf.sav')


import questionAnswering.squadAnalysis

# from bert_serving.client import BertClient
# from scipy.spatial.distance import cosine
#
# bc = BertClient(check_length=False)
#
#
# while True:
#     s = input("s: ")
#     print(get_word_encodings(s))
    # sVec = bc.encode([s])
    # sWords = ['CLS'] + s.split() + ['SEP']
    # for i, wordVec in enumerate(sVec[0]):
    #     if not wordVec[0]==0:
    #         print(sWords[i], wordVec)
    #         print(np.sum(wordVec))
