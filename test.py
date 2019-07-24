# from utils.objectSaver import load
#
# from clustering.clusterStruct import ProjectableCluster
# from vectorizers.docVecs import vectorize_doc
# from utils.cleaner import clean_text

from documentRetrieval.questionReading import read_question_dataset
PATH = 'data/inData/natural_questions/v1.0/train'
read_question_dataset(PATH)

# nounSet = set()
# with open('data/inData/nouns.txt', 'r') as nounFile:
#     for line in nounFile:
#         nounSet.add(clean_text(line))
# cluster = ProjectableCluster(nounSet, outPath='data/outData/ProjectableCluster.sav')

# cluster = ProjectableCluster(inPath='data/outData/ProjectableCluster.sav')
# import os
# textList = []
# path = 'data/inData/imdbData/train/pos'
# for i, file in enumerate(os.listdir(path)[:500]):
#     with open(f'{path}/{file}', 'r') as imdbFile:
#         textList.append(imdbFile.read())
#
# cluster = ProjectableCluster(textList, outPath='data/outData/imdbCluster.sav')

# while True:
#     search = input("Search: ")
#     newVec = vectorize_doc(search)
#     print(cluster.find_nearest_via_centroid(newVec, n=5))
