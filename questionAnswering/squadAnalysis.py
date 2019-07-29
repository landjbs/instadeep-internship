import re
import json
import numpy as np
# from tqdm import tqdm

answerMetrics = []
with open('data/inData/train-v2.0.json') as squadFile:
    for categorty in json.load(squadFile)['data']:
        for paragraph in categorty['paragraphs']:
            paragraphText = paragraph['context']
            paraSents = re.split(r'[.?!]', paragraphText)
            print(paraSents)
            for qas in paragraph['qas']:
                question = qas['question']
                answerList = qas['answers']
                if not answerList==[]:
                    answerText = answerList[0]['text']
                    print(answerText)
