# from vectorizers.datasetVectorizer import vectorize_folderList
# folderList = ['data/inData/imdbData/train/pos', 'data/inData/imdbData/train/neg']
# vectorize_folderList(folderList, numFiles=5000, outPath='data/outData/imdbTrain5000.sav')

# from supervised.folderPrediction import train_folder_model
# train_folder_model('data/outData/imdbTrain5000.sav', outPath='data/outData/models/folderModel5000.sav')

from supervised.sentimentAnalyzer import analyze_text

while True:
    text = input('test: ')
    analyze_text(text)
