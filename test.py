
import questionAnswering.squadAnalysis



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
