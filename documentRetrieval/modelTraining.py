import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def train_retrieval_model(inPath, numWrong=3, outPath=None):
    ### import and modify dataframe ###
    questionsDf = pd.read_pickle(inPath)
    questionsDf = questionsDf.drop(columns=['longAnswerStarts'])

    ### shuffle dataframe such that each question has numWrong wrong answers ###
    rowNumList = list([i for i, _ in enumerate(questionsDf.iterrows())])

    shuffledList = []

    for baseRow in questionsDf.iterrows():
        baseRowIndex = baseRow[0]
        baseRowInfo = baseRow[1]

        print(f"Shuffling: {baseRowIndex}", end='\r')

        basePageUrl      =  baseRowInfo[0]
        baseQuestionText =  baseRowInfo[1]
        baseQuestionVec  =  baseRowInfo[2]
        baseTextVec      =  baseRowInfo[3]
        baseTitleVec     =  baseRowInfo[4]

        baseDict = {'pageUrl':        basePageUrl,
                    'questionText':   baseQuestionText,
                    'questionVec':    baseQuestionVec,
                    'textVec':        baseTextVec,
                    'titleVec':       baseTitleVec,
                    'score':          1}

        shuffledList.append(baseDict)

        # choose two rows at random from the dataset
        possibleRows = rowNumList.copy()
        del possibleRows[baseRowIndex]
        otherRows = np.random.choice(possibleRows, size=numWrong)

        for otherRowIndex in otherRows:
            otherRowInfo = questionsDf.iloc[otherRowIndex]

            otherRowUrl       =  otherRowInfo[0]
            otherRowTextVec   =  otherRowInfo[3]
            otherRowTitleVec  =  otherRowInfo[4]

            otherRowDict = baseDict.copy()
            otherRowDict.update({'pageUrl':    otherRowUrl,
                                'textVec':   otherRowTextVec,
                                'titleVec':  otherRowTitleVec,
                                'score':     0})
            shuffledList.append(otherRowDict)
            del otherRowDict
        del baseDict

    shuffledDf = pd.DataFrame(shuffledList)

    ### simplify dataframe to store distance between textVec and questionVec ###
    dataList = []
    for row in shuffledDf.iterrows():
        rowInfo = row[1]
        questionVec, textVec = rowInfo[2], rowInfo[4]
        vecDifference = np.subtract(questionVec, textVec)
        dataDict = {dim:scalar for dim,scalar in enumerate(vecDifference)}
        dataDict.update({'score':rowInfo[3]})
        dataList.append(dataDict)

    dataDf = pd.DataFrame(dataList)

    ### format dataframe for training ###
    # reindex the data
    dataDf = dataDf.reindex(np.random.permutation(dataDf.index))

    targets = dataDf['score']
    features = dataDf.drop(columns=['score'])

    # normalize the features
    normedFeatures = pd.DataFrame(StandardScaler().fit_transform(X=features))
    normedFeatures.describe()

    ### train model ###
    model = Sequential()
    # initialize bias weights
    scoreWeights = {0 : 0.25,
                    1 : 0.75}
    # input layer
    model.add(Dense(30, activation='relu', input_dim=(normedFeatures.shape)[1]))
    # first hidden layer
    model.add(Dense(100, activation='relu'))
    # second hidden layer
    model.add(Dense(100, activation='relu', kernel_regularizer=l2(0.01)))
    # output layer
    model.add(Dense(1, activation='sigmoid'))
    # compile and train
    model.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])
    model.fit(features, targets, validation_split=0.1, epochs=15, class_weight=scoreWeights)
    # save model to outPath if given
    if outPath:
        model.save(outPath)
    return model
