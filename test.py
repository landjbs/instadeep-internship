from questionAnswering.squadAnalysis import read_squad_dataset

read_squad_dataset(squadPath='data/inData/train-v2.0.json',
                    paraDepth=1,
                    picklePath='data/outData/squadDataFrame.sav',
                    csvPath='data/outData/squadDataFrameBACKUP.csv')
# import questionAnswering.kerasModel
