import numpy as np
import pandas as pd
import tensorflow as tf
from functools import reduce

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Bidirectional


df = pd.read_pickle('data/outData/squadDataFrame.sav')

features, targets = df['features'], df['targets']

print(targets)

featureArray = np.array([feature for feature in features])
# targetArray = np.array([np.array(target) for target in targets])

for target in targets:
    print(len(np.array(target)))

print(f'{"-"*80}\n{targetArray.shape}\n{"-"*80}')

model = Sequential()
model.add(Bidirectional(LSTM(10, return_sequences=True),
                                input_shape=(featureArray.shape[1],
                                            featureArray.shape[2])))
model.add(Bidirectional(LSTM(10)))
model.add(Dense(5))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# # With custom backward layer
# model = Sequential()
# forward_layer = LSTM(10, return_sequences=True)
# backard_layer = LSTM(10, activation='relu', return_sequences=True,
#                     go_backwards=True)
# model.add(Bidirectional(forward_layer, backward_layer=backward_layer,
#                        input_shape=(5, 10)))

model.add(Dense(402))
model.add(Activation('softmax'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop')

model.fit(featureArray, targetArray)
