from questionAnswering.questionFormat import create_fake_queries
from questionAnswering.questionAnalysis import build_question_database

fakeDf = create_fake_queries(6215, outPath='data/outData/fakeQueryDf'))

realDf = build_question_database()
