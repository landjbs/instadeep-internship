"""
Modularized prediction of folder using BERT dimensions as features.
Takes dataframes generated by dataset vectorizer as inputs
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.regularizers import l2

def train_folder_model(inPath, testFraction=0.1, epochs=10, outPath=None):
    """
        -inPath:        Path at which dataframe has been pickled
        -testFraction:  The fraction of the data that will be used for
                            testing rather than training.
        -epochs:        The number of epochs for which to train the model.
        -outPath:       Path to which to save the model.
    """
    # load dataframe from inPath
    rawDf = pd.read_pickle(inPath)

    # drop file and folder columns and save folder column as targets
    vecDf = rawDf.drop(columns=['file', 'folder'])
    rawTargets = list(rawDf['folder'])

    # normalize z-score of vecDf dimensions
    normalizedVecDf = pd.DataFrame(StandardScaler().fit_transform(X=vecDf))
    print(f'Normalized Dataframe Head:\n{normalizedVecDf.describe().head()}')

    # one-hot encode targets
    targetDict = {target:i for i, target in enumerate(set(rawTargets))}
    numericTargets = [targetDict[target] for target in rawTargets]
    encodedTargets = to_categorical(numericTargets)

    # train_test_split the data
    x_train, y_train, x_test, y_test = train_test_split(normalizedVecDf,
                                                        encodedTargets,
                                                        test_size=testFraction)

    # model architecture
    model = Sequential()
    model.add(Dense(100, activation='relu', input_dim=((x_train.shape)[1])))
    model.add(Dense(100, activation='relu', kernel_regularizer=l2(0.01)))
    # binary output layer with sigmoid activation
    model.add(Dense(1, activation='sigmoid'))
    # compile the model
    model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
    # train the model
    model.fit(x_train, y_train, epochs=epochs)

    # display test metrics if test data is allocated
    if (testFraction != 0):
        modelMetrics = model.evaluate(x_test, y_test)
        print('Evaluation:')
        for i, metric in (model.metrics_names):
            print(f'\t{metric}: {modelMetrics[i]}')

    # save model if outPath is given
    if outPath:
        model.save(outPath)
    return model
