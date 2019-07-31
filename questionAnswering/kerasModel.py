import numpy as np
import pandas as pd
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Bidirectional


df = pd.read_pickle('data/outData/squadDataFrame.sav')
print(df.shape)
print(df.head())

features, targets = df['features'], df['targets']

model = Sequential()
model.add(Bidirectional(LSTM(10, return_sequences=True), input_shape=(features[0].shape[0], features[0].shape[1])))
model.add(Bidirectional(LSTM(10)))
model.add(Dense(5))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# With custom backward layer
# model = Sequential()
# forward_layer = LSTM(10, return_sequences=True)
# backard_layer = LSTM(10, activation='relu', return_sequences=True,
#                     go_backwards=True)
# model.add(Bidirectional(forward_layer, backward_layer=backward_layer,
#                        input_shape=(5, 10)))


model.add(Dense(1))
model.add(Activation('softmax'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop')

model.fit(features, targets)
