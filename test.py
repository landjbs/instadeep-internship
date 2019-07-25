# from utils.objectSaver import load
#
# from clustering.clusterStruct import ProjectableCluster
# from vectorizers.docVecs import vectorize_doc
# from utils.cleaner import clean_text


# from documentRetrieval.questionReading import read_question_dataset
# PATH = 'data/inData/natural_questions/v1.0/train'
# x = read_question_dataset(PATH, n=2500, outPath='data/outData/questionsDf2500.sav')


# from documentRetrieval.modelTraining import train_retrieval_model
# retrievalModel = train_retrieval_model(inPath='data/outData/questionsDf2500.sav',
#                                         numWrong=3,
#                                         outPath='data/outData/models/documentRetrievalModel2500.sav')

from documentRetrieval.retreiverAPI import Retreiver
wikiSearcher = Retreiver('data/inData/wikipedia_utf8_filtered_20pageviews.csv', n=10)

while True:
    search = input("Search: ")
    wikiSearcher.retreive(search)
