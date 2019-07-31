import pandas as pd

df = pd.read_csv('data/outData/squadDataBACKUP.sav')

print(df)

from questionAnswering.squadAnalysis import read_squad_dataset

read_squad_dataset(squadPath='data/inData/train-v2.0.json',
                    paraDepth=2,
                    picklePath='data/outData/squadDataFrame.sav',
                    csvPath='data/outData/squadDataFrameBACKUP.csv')

# from bert_serving.client import BertClient
# from nltk.tokenize import word_tokenize
#
#
# bc = BertClient(check_length=True)
#
# while True:
#     sent = input('sent: ')
#     tokens = word_tokenize(sent)
#     sentVec = bc.encode([tokens], is_tokenized=True)
#     sentVec = list(filter(lambda l:l[0]!=0, sentVec[0]))
#     print([(tokens[i], wordVec) for i, wordVec in enumerate(sentVec)])
