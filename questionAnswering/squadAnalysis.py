import re
import json
import numpy as np
from tqdm import tqdm
from vectorizers.docVecs import get_word_encodings

def clean_text(text):
    decimalSub = re.sub(r'(?<=.)\.(?=.)', ',', text)
    return re.sub(r'e.g.', "", decimalSub)


answerMetrics = []
with open('data/inData/train-v2.0.json') as squadFile:
    for categorty in json.load(squadFile)['data']:
        print(f"Category: {categorty['title']}")
        for paragraph in tqdm(categorty['paragraphs']):
            paragraphText = paragraph['context'].lower()
            try:
                paragraphEmbeddings = get_word_encodings(paragraphText)
            except:
                pass
            # for qas in paragraph['qas']:
            #     question = qas['question'].lower()
            #     answerList = qas['answers']
            #     if not answerList==[]:
            #         answerText = clean_text(answerList[0]['text'].lower())
            #         isIn = False
            #         for sent in paraSents:
            #             if answerText in sent:
            #                 isIn = True
            #                 numIn += 1
            #     else:
            #         pass
