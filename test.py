# Read data
# from questionAnswering.squadAnalysis import read_squad_dataset
# read_squad_dataset(squadPath='data/inData/train-v2.0.json',
#                     paraDepth=2,
#                     pickleFolder='data/outData/squadDataFrames',)

# Train model
from questionAnswering.kerasModel import train_answering_lstm
model = train_answering_lstm(folderPath='data/outData/squadDataFrames',
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
Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/; born September 4, 1981)[4] is an American singer, actress, songwriter, record producer, director, model, dancer, fashion designer and businesswoman. Born and raised in Houston, Texas, Beyoncé performed in various singing and dancing competitions as a child. She rose to fame in the late 1990s as lead singer of the R&B girl-group Destiny's Child. Managed by her father, Mathew Knowles, the group became one of the best-selling girl groups in history. Their hiatus saw Beyoncé's theatrical film debut in Austin Powers in Goldmember (2002) and the release of her first solo album, Dangerously in Love (2003). The album established her as a solo artist worldwide, debuting at number one on the US Billboard 200 chart and earning her five Grammy Awards.[5] The album also featured the US Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy"."""

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
    predictions = [(p, (i - (questionArray.shape[0]) - 1)) for i, p in enumerate(model.predict(expandedFeatures)[0])]

    predictions.sort(reverse=True)

    maxLocs = [elt[1] for elt in predictions[:10]]
    print(maxLocs)

    for predLoc in maxLocs:
        print(contextTokens[predLoc])

    # maxPredictionLoc = predictions.index(max(predictions))
    # actualMaxLoc = maxPredictionLoc -
    # print(actualMaxLoc)
    # print(contextTokens[actualMaxLoc])
    # print(predictions)

    # plt.plot(predictions)
    # plt.show()
