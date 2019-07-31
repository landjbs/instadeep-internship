import re
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from vectorizers.docVecs import get_word_encodings
from nltk.tokenize import word_tokenize, sent_tokenize
from bert_serving.client import BertClient

bc = BertClient(check_length=True)

MAX_LEN = 390

def filter_text_vec(textVec, numWords):
    """
    Returns np.array of shape (numWords, numDims)
        -textVec:   the contextual vector of each word in the text
        -numWords:  the number of word vectors allowed in the final array
    """
    return np.array([wordVec for i, wordVec in enumerate(textVec)
                    if i <= numWords])


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


def read_squad_dataset(squadPath, paraDepth=2, picklePath=None, csvPath=None):
    """
    Reads SQuAD dataset from json file into LSTM-ready dataframe mapping a
    feature array of the contextual embedding of each token in a text with
    the embedding of each token in a question concatenated to the front and
    a target array of len=features.shape[0] where tokens within the answer
    span of the question have a 1 and the rest have a 0.
        -squadPath:     File from which to read the squad data
        -paraDepth:     Number of questions from each paragraph to analyze
        -picklePath:    Path to which to save the final dataframe as pickle
        -csvPath:       Path to which to save the final dataframe as csv (backup)
    """
    dataList = []
    with open(squadPath) as squadFile:
        for categorty in json.load(squadFile)['data']:
            print(f"Category: {categorty['title']}")

            for i, paragraph in enumerate(tqdm(categorty['paragraphs'])):
                try:
                    assert (i < paraDepth), f"Paragraph Num Exceeded at paragraph number {i}."

                    # convert paragraph into filtered array of contextual word vecs
                    paragraphText = paragraph['context'].lower()
                    paragraphTokens = word_tokenize(paragraphText)

                    assert (len(paragraphTokens)<=MAX_LEN), f"Paragraph has {len(paragraphTokens)} tokens; cannot be more than {MAX_LEN}."

                    paragraphVec = bc.encode([paragraphTokens], is_tokenized=True)[0]
                    paragraphArray = filter_text_vec(paragraphVec)

                    for qas in tqdm(paragraph['qas'], leave=False, ncols=70):
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

    if picklePath:
        try:
            dataframe.to_pickle(picklePath)
        except Exception as e:
            print(f'PICKLE ERROR: {e}')
    if csvPath:
        try:
            dataframe.to_csv(csvPath)
        except Exception as e:
            print(f'CSV ERROR: {e}')

    return dataframe
