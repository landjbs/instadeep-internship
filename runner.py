from vectorizers.datasetVectorizer import vectorize_folderList
# from supervised.folderPrediction import train_folder_model

folderList = ['data/inData/imdbData/test/neg', 'data/inData/imdbData/test/pos']
vectorize_folderList(folderList, numFiles=1000, cleanFiles=False, outPath="data/outData/imdbTrain.sav")

# train_folder_model(inPath='data/outData/imdbTrain.sav', outPath='data/outData/models/imdbModel_Small.sav')
