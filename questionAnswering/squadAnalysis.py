import re
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from vectorizers.docVecs import get_word_encodings
from nltk.tokenize import word_tokenize, sent_tokenize
from bert_serving.client import BertClient

bc = BertClient(check_length=True)

MAX_LEN = 490

def filter_text_vec(textVec):
    """
    filters out empty vectors of buffer words in text vec
    and returns np.array of shape (numWords, numDims)
    """
    return np.array([wordVec for wordVec in textVec if not (wordVec[0]==0)])


def make_target_list(answerTokens, paragraphTokens, questionTokens):
    """
    Makes a list of targets for training where answer tokens have score of
    1 and everything else (including the question tokens) have score of 0
    """
    answerLen = len(answerTokens)
    firstAnswerWord = answerTokens[0]
    for i, word in enumerate(paragraphTokens):
        if (word == firstAnswerWord):
            if all((word in answerTokens) for word
                    in paragraphTokens[i : (i + answerLen)]):
                answerStart, answerEnd = i, (i + answerLen)
    paragraphTargets = [1 if i in range(answerStart, answerEnd) else 0
                        for i in range(len(paragraphTokens))]
    return ([0 for _ in range(len(questionTokens))] + paragraphTargets)


dataList = []
with open('data/inData/train-v2.0.json') as squadFile:
    for categorty in json.load(squadFile)['data']:
        print(f"Category: {categorty['title']}")

        for paragraph in categorty['paragraphs']:
            try:
                # convert paragraph into filtered array of contextual word vecs
                paragraphText = paragraph['context'].lower()
                paragraphTokens = word_tokenize(paragraphText)

                assert (len(paragraphTokens)>=MAX_LEN), f"Paragraph has {len(paragraphTokens)} tokens— cannot be more than {MAX_LEN}."

                paragraphVec = bc.encode([paragraphTokens], is_tokenized=True)[0]
                paragraphArray = filter_text_vec(paragraphVec)

                for qas in paragraph['qas']:
                    # convert question into filtered array of conxtual word vecs
                    question = qas['question'].lower()
                    questionTokens = word_tokenize(question)
                    questionVec = bc.encode([questionTokens], is_tokenized=True)[0]
                    questionArray = filter_text_vec(questionVec)

                    answerList = qas['answers']

                    if not answerList==[]:
                        answerText = answerList[0]['text'].lower()
                        answerTokens = word_tokenize(answerText)
                        targetList = make_target_list(answerTokens, paragraphTokens, questionTokens)
                    else:
                        targetList = [0 for _ in range(len(questionTokens) + len(paragraphTokens))]

                    featureArray = np.concatenate([paragraphArray, questionArray], axis=0)

                    dataList.append({'features':featureArray, 'targets':targetList})
            except:
                pass

dataframe = pd.DataFrame(dataList)

dataframe.to_pickle('data/outData/squadDataFrame.sav')
