# from documentRetrieval.questionReading import read_question_dataset
# PATH = 'data/inData/natural_questions/v1.0/train'
# x = read_question_dataset(PATH, n=2500, outPath='data/outData/questionsDf2500.sav')

# from questionAnswering.questionAnalysis import build_question_database
# PATH = 'data/inData/natural_questions/v1.0/train'
# x = build_question_database(PATH, n=2500, outPath='data/outData/answeringDf2500.sav')


import pandas as pd

x = pd.read_pickle('data/outData/answeringDf2500.sav')
print(x)
