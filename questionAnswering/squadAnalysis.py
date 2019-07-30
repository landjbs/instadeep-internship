import re
import json
import numpy as np
from tqdm import tqdm
from vectorizers.docVecs import get_word_encodings
from nltk.tokenize import word_tokenize, sent_tokenize
from bert_serving.client import BertClient

bc = BertClient(check_length=True)

dataList = []
with open('data/inData/train-v2.0.json') as squadFile:
    for categorty in json.load(squadFile)['data']:
        print(f"Category: {categorty['title']}")

        for paragraph in (categorty['paragraphs']):
            paragraphText = paragraph['context'].lower()

            sentences = sent_tokenize(paragraphText)
            tokenizedSents = [word_tokenize(sent) for sent in sentences]
            sentVecs = bc.encode(tokenizedSents, is_tokenized=True)

            wordEmbeddings = []
            for sentNum, wordVecs in enumerate(sentVecs):
                for wordNum, wordVec in enumerate(wordVecs):
                    if not (wordVec[0]==0):
                        curWord = tokenizedSents[sentNum][wordNum-1]
                        wordEmbeddings.append([curWord] + [scalar for scalar in wordVec])

            wordEmbeddings = np.array(wordEmbeddings)

            # scoring [inAnswerSpan, wordVec dims-->]
            for qas in paragraph['qas']:
                question = re.sub("\\?", "", qas['question'].lower())
                questionWords = word_tokenize(question)
                wordVecs = bc.encode([questionWords], is_tokenized=True)[0]
                questionEmbeddings = []
                for wordVec in wordVecs:
                    if not wordVec[0]==0:
                        questionEmbeddings.append([0] + [scalar for scalar in wordVec])

                questionEmbeddings = np.array(questionEmbeddings)

                data = np.concatenate((questionEmbeddings, wordEmbeddings), axis=0)
                print(f"Data: {data.shape}")

                # print(f'questionVec: {len(questionVec)}')
                answerList = qas['answers']
                if answerList==[]:
                    answerTokens = None
                    wordEmbeddings[:, 0] = np.zeros(wordEmbeddings.shape[0])
                else:
                    answerText = answerList[0]['text'].lower()
                    answerWords = word_tokenize(answerText)
                    answerLen = len(answerWords)
                    print(answerText)
                    for row in wordEmbeddings:
                        if row[0] in answerWords:
                            print(row[0])
