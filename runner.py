from vectorizers.datasetVectorizer import vectorize_folderList
from supervised.folderPrediction import train_folder_model

folderList = ['imdbData/test/neg', 'imdbData/test/pos']
vectorize_folderList(folderList, numFiles=1000, cleanFiles=False, outPath="imdbTrain.sav")
