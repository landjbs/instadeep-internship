import re
import json
import numpy as np
from tqdm import tqdm
from vectorizers.docVecs import get_word_encodings
from nltk.tokenize import word_tokenize, sent_tokenize

def clean_text(text):
    decimalSub = re.sub(r'(?<=.)\.(?=.)', ',', text)
    return re.sub(r'e.g.', "", decimalSub)


answerMetrics = []
with open('data/inData/train-v2.0.json') as squadFile:
    for categorty in json.load(squadFile)['data']:
        print(f"Category: {categorty['title']}")
        for paragraph in (categorty['paragraphs']):
            paragraphText = paragraph['context'].lower()
            # paragraphEmbeddings = get_word_encodings(paragraphText)


            # for qas in paragraph['qas']:
            #     question = qas['question'].lower()
            #     answerList = qas['answers']
            #     if answerList==[]:
            #         answerTokens = None
            #     else:
            #         print(question)
            #         answerText = answerList[0]['text'].lower()
            #         print(paragraphText.index(answerText))
            #         print(answerList[0]['answer_start'])
            # except:
            #     pass
