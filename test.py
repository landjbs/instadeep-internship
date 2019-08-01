# Read data
from questionAnswering.squadAnalysis import read_squad_dataset
df = read_squad_dataset(squadPath='data/inData/train-v2.0.json',
                    paraDepth=1,
                    pickleFolder='data/outData/squadDataFrames',)

# Train model
from questionAnswering.kerasModel import train_answering_lstm
model = train_answering_lstm(filePath='data/outData/squadDataFrame.sav',
                            outPath='data/outData/models/answeringModel.sav')


# Test model
import numpy as np
from bert_serving.client import BertClient
from keras.models import load_model
from nltk.tokenize import word_tokenize, sent_tokenize
from scipy.special import softmax

bc = BertClient()

model = load_model('data/outData/models/answeringModel.sav')


def filter_text_vec(textVec, numWords):
    """
    Returns np.array of shape (numWords, numDims)
        -textVec:   the contextual vector of each word in the text
        -numWords:  the number of word vectors allowed in the final array
    """
    return np.array([wordVec for i, wordVec in enumerate(textVec)
                    if i <= numWords])


context = """
A good article (GA) is an article that meets a core set of editorial standards but is not featured article quality.
Good articles meet the good article criteria, passing through the good article nomination process successfully.
They are well written, contain factually accurate and verifiable information, are broad in coverage, neutral in point of view, stable,
and illustrated, where possible, by relevant images with suitable copyright licenses.
Good articles do not have to be as comprehensive as featured articles, but they should not omit any major facets of the topic:
a comparison of the criteria for good and featured articles describes further differences.
"""

contextTokens = word_tokenize(context)
contextVec = bc.encode([contextTokens], is_tokenized=True)[0]
contextArray = filter_text_vec(contextVec, 390)
print(contextArray.shape)

import matplotlib.pyplot as plt

while True:
    question = input('Question: ')
    questionTokens = word_tokenize(question)
    questionVec = bc.encode([questionTokens], is_tokenized=True)[0]
    questionArray = filter_text_vec(questionVec, 12)
    featureArray = np.concatenate([questionArray, contextArray], axis=0)
    print(f'Shape: {featureArray.shape}')
    expandedFeatures = np.expand_dims(featureArray, axis=0)
    print(expandedFeatures.shape)
    predictions = [elt for elt in model.predict(expandedFeatures)[0]]
    maxPredictionLoc = predictions.index(max(predictions))
    actualMaxLoc = maxPredictionLoc - (questionArray.shape[0]) - 1
    print(actualMaxLoc)
    print(contextTokens[actualMaxLoc])
    # print(predictions)
    # plt.plot(predictions)
    # plt.show()
