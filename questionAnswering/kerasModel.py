import numpy as np
import pandas as pd
import tensorflow as tf
from os import listdir
from functools import reduce
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Bidirectional, ConvLSTM2D, Masking
from keras.utils import plot_model


def train_answering_lstm(folderPath, outPath=None):
    """
    Trains bidirectional LSTM model on dataframe of feature arrays and
    target vectors in dataframe pickled at filePath.
        -folderPath:    Path to folder under which the dataframe is tableted
        -outPath:       Path to which to save the trained model
    """

    # build dataframe form tablets under folderPath
    tabletList = []
    for file in listdir(folderPath):
        if file.endswith('.sav'):
            tablet = pd.read_pickle(f'{folderPath}/{file}', compression='gzip')
            tabletList.append(tablet)

    dataframe = pd.concat(tabletList)

    features, targets = dataframe['features'], dataframe['targets']

    # reshape the feature and target arrays
    featureArray = np.array([feature for feature in features])
    targetArray = np.array([np.array(target) for target in targets])

    # # Display
    # plt.plot(np.sum(targetArray, axis=0))
    # plt.xlabel('Token Num')
    # plt.ylabel('Number of Times in Span')
    # plt.show()


    maskArray = np.zeros(featureArray.shape[2])
    print(f'Mask: {maskArray.shape}')

    # model architecture
    model = Sequential()
    model.add(Masking(mask_value=maskArray))
    model.add(Bidirectional(LSTM(40), input_shape=(featureArray.shape[1],
                                                featureArray.shape[2])))
    # model.add(Bidirectional(LSTM(400)))

    # # With custom backward layer
    # model = Sequential()
    # forward_layer = LSTM(10, return_sequences=True)
    # backard_layer = LSTM(10, activation='relu', return_sequences=True,
    #                     go_backwards=True)
    # model.add(Bidirectional(forward_layer, backward_layer=backward_layer,
    #                        input_shape=(5, 10)))

    model.add(Dense(targetArray.shape[1]))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    from keras.utils import plot_model
    plot_model(model, to_file='model.png')

    # model training
    model.fit(featureArray, targetArray, batch_size=200, epochs=50)

    if outPath:
        model.save(outPath)

    plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    return model
