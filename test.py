# from questionAnswering.questionAnalysis import build_question_database
# PATH = 'data/inData/natural_questions/v1.0/train'
# x = build_question_database(PATH, n=2500, outPath='data/outData/answeringDf2500.sav')
import numpy as np
from keras.models import load_model
from vectorizers.docVecs import vectorize_doc
model = load_model('data/outData/models/queryAnalysisModel.sav')
while True:
    search = input('search: ')
    searchVec = vectorize_doc(search)
    print(model.predict(np.expand_dims(searchVec, axis=0)))
