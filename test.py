# Read data
# from questionAnswering.squadAnalysis import read_squad_dataset
# read_squad_dataset(squadPath='data/inData/train-v2.0.json',
#                     paraDepth=12,
#                     pickleFolder='data/outData/squadDataFrames',)

# Train model
# from questionAnswering.kerasModel import train_answering_lstm
# model = train_answering_lstm(folderPath='data/outData/squadDataFrames',
#                             outPath='data/outData/models/answeringModel.sav')


# # Test model
# import numpy as np
# from bert_serving.client import BertClient
# from keras.models import load_model
# from nltk.tokenize import word_tokenize, sent_tokenize
# from scipy.special import softmax
#
# bc = BertClient()
#
# model = load_model('data/outData/models/answeringModel.sav')
#
#
# def filter_text_vec(textVec, numWords):
#     """
#     Returns np.array of shape (numWords, numDims)
#         -textVec:   the contextual vector of each word in the text
#         -numWords:  the number of word vectors allowed in the final array
#     """
#     return np.array([wordVec for i, wordVec in enumerate(textVec)
#                     if i <= numWords])
#
# context = """ Architecturally, the school has a Catholic character. Atop the Main Building\’s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend “Venite Ad Me Omnes”. Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary """
#
# contextTokens = word_tokenize(context)
# contextVec = bc.encode([contextTokens], is_tokenized=True)[0]
# contextArray = filter_text_vec(contextVec, 390)
# print(contextArray.shape)
#
# import matplotlib.pyplot as plt
#
# while True:
#     question = input('Question: ')
#     questionTokens = word_tokenize(question)
#     questionVec = bc.encode([questionTokens], is_tokenized=True)[0]
#     questionArray = filter_text_vec(questionVec, 12)
#     print(questionArray)
#     featureArray = np.concatenate([questionArray, contextArray], axis=0)
#     print(featureArray.shape)
#     print(f'Shape: {featureArray.shape}')
#     expandedFeatures = np.expand_dims(featureArray, axis=0)
#     print(expandedFeatures.shape)
#     predictions = [(p, (i - (questionArray.shape[0]) - 1)) for i, p in enumerate(model.predict(expandedFeatures)[0])]
#
#     predictions.sort(reverse=True)
#
#     maxLocs = [elt[1] for elt in predictions[:10]]
#     print(maxLocs)
#
#     # maxPredictionLoc = predictions.index(max(predictions))
#     # actualMaxLoc = maxPredictionLoc -
#     # print(actualMaxLoc)
#     # print(contextTokens[actualMaxLoc])
#     # print(predictions)
#
#     # plt.plot(predictions)
#     # plt.show()
#
#     for predLoc in maxLocs:
#         print(contextTokens[predLoc])
