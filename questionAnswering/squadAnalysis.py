import re
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from vectorizers.docVecs import get_word_encodings
from nltk.tokenize import word_tokenize, sent_tokenize
from bert_serving.client import BertClient

bc = BertClient(check_length=True)


def filter_text_vec(textVec, numWords):
    """
    Returns np.array of shape (numWords, numDims)
        -textVec:   the contextual vector of each word in the text
        -numWords:  the number of word vectors allowed in the final array
    """
    return np.array([wordVec for i, wordVec in enumerate(textVec)
                    if i <= numWords])

def make_target_list(answerTokens, paragraphTokens, questionLen, paraLen):
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
                        for i in range(paraLen)]
    return ([0 for _ in range(questionLen)] + paragraphTargets)


def read_squad_dataset(squadPath, paraDepth=2, paraMax=390, questionMax=12, picklePath=None, csvPath=None):
    """
    Reads SQuAD dataset from json file into LSTM-ready dataframe mapping a
    feature array of the contextual embedding of each token in a text with
    the embedding of each token in a question concatenated to the front and
    a target array of len=features.shape[0] where tokens within the answer
    span of the question have a 1 and the rest have a 0.
        -squadPath:     File from which to read the squad data
        -paraDepth:     Number of questions from each paragraph to analyze
        -paraMax:       Max number of tokens that a paragraph can have to be analyzed
        -questionMax:   Max number of tokens that a question can have to be analyzed
        -picklePath:    Path to which to save the final dataframe as pickle
        -csvPath:       Path to which to save the final dataframe as csv (backup)
    """

    dataList = []
    with open(squadPath) as squadFile:
        print(f"{'-'*30}[ Analyzing SQuAD Dataset ]{'-'*30}")
        for categorty in tqdm(json.load(squadFile)['data']):
            # print(f"Category: {categorty['title']}")

            for i, paragraph in enumerate(tqdm(categorty['paragraphs'], leave=False, ncols=120)):
                try:
                    assert (i < paraDepth), f"Paragraph Num Exceeded at paragraph number {i}."

                    # convert paragraph into filtered array of contextual word vecs
                    paragraphText = paragraph['context'].lower()
                    paragraphTokens = word_tokenize(paragraphText)

                    assert (len(paragraphTokens)<=paraMax), f"Paragraph has {len(paragraphTokens)} tokens; cannot be more than {paraMax}."

                    paragraphVec = bc.encode([paragraphTokens], is_tokenized=True)[0]
                    paragraphArray = filter_text_vec(paragraphVec, paraMax)

                    for qas in tqdm(paragraph['qas'], leave=False, ncols=70):
                        try:
                            # convert question into filtered array of conxtual word vecs
                            question = qas['question'].lower()
                            questionTokens = word_tokenize(question)

                            assert (len(questionTokens)<=questionMax), f"Question has {len(questionTokens)} tokens; cannot be more than {questionMax}."

                            questionVec = bc.encode([questionTokens], is_tokenized=True)[0]
                            questionArray = filter_text_vec(questionVec, questionMax)

                            answerList = qas['answers']

                            if not answerList==[]:
                                answerText = answerList[0]['text'].lower()
                                answerTokens = word_tokenize(answerText)
                                targetList = make_target_list(answerTokens,
                                                            paragraphTokens,
                                                            questionArray.shape[0],
                                                            paragraphArray.shape[0])
                            else:
                                targetList = [0 for _ in range(questionArray.shape[0] + paragraphArray.shape[0])]

                            featureArray = np.concatenate([questionArray, paragraphArray], axis=0)

                            dataList.append({'features':featureArray, 'targets':targetList})
                        except:
                            pass
                except:
                    pass

    dataframe = pd.DataFrame(dataList)

    # save dataframe if prompted
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
