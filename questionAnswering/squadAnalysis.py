import re
import json
import numpy as np
from tqdm import tqdm

def clean_text(text):
    decimalSub = re.sub(r'(?<=.)\.(?=.)', ',', text)
    return re.sub(r'e.g.', "", decimalSub)


answerMetrics = []
with open('data/inData/train-v2.0.json') as squadFile:
    for categorty in (json.load(squadFile)['data']):
        for paragraph in categorty['paragraphs']:
            paragraphText = clean_text(paragraph['context'])
            paraSents = [sent.strip().lower()
                        for sent in re.split(r'[.?!]', paragraphText)
                        if not sent==""]
            for qas in paragraph['qas']:
                question = qas['question'].lower()
                answerList = qas['answers']
                if not answerList==[]:
                    answerText = clean_text(answerList[0]['text'].lower())
                    isIn = False
                    for sent in paraSents:
                        if answerText in sent:
                            isIn = True
                            numIn += 1
                    if not isIn:
                        # print(f'{answerText}\n{paraSents}')
                        notIn += 1
                else:
                    pass
