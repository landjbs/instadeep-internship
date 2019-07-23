from vectorizers.datasetVectorizer import vectorize_folderList
from supervised.folderPrediction import train_folder_model

# vectorize_folderList(['test1', 'test2'], cleanFiles=True, outPath="out.test")
train_folder_model('out.test')
