from questionAnswering.squadAnalysis import read_squad_dataset

read_squad_dataset(squadPath='data/inData/train-v2.0.json',
                    paraDepth=1,
                    picklePath='data/outData/squadDataFrame.sav',
                    csvPath='data/outData/squadDataFrameBACKUP.csv')

# from questionAnswering.kerasModel import train_answering_lstm
#
# train_answering_lstm('data/outData/squadDataFrame.sav',
#                     outPath='data/outData/models/answeringModel.sav')
