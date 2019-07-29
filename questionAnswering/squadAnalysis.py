import re
import json
import numpy as np
from tqdm import tqdm

def clean_text(text):
    return re.sub(r'[(\[].+[\])]', "", text)

answerMetrics = []
with open('data/inData/train-v2.0.json') as squadFile:
    for categorty in (json.load(squadFile)['data']):
        for paragraph in categorty['paragraphs']:
            paragraphText = clean_text(paragraph['context'])
            paraSents = re.split(r'[.?!]', paragraphText)
            for qas in paragraph['qas']:
                question = qas['question']
                answerList = qas['answers']
                if not answerList==[]:
                    answerText = answerList[0]['text']
                    isIn = False
                    for sent in paraSents:
                        if answerText in sent:
                            isIn = True
                    if not isIn:
                        print(answerText)
                else:
                    pass
