import re
import numpy as np
from termcolor import colored
from flashtext import KeywordProcessor
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

import vectorizers.docVecs as docVecs


def build_keyword_processor(knowledgeSet):
    """ Builds flashtext matcher for words in knowledgeSet iterable """
    # initialize flashtext KeywordProcessor
    keywordProcessor = KeywordProcessor(case_sensitive=False)
    for i, keyword in enumerate(knowledgeSet):
        print(f"\tBuilding keywordProcessor: {i}", end="\r")
        keywordProcessor.add_keyword(keyword)
    print("\keywordProcessor Built")
    return keywordProcessor


def vectorize_masked_tokens(document, maskToken='', keywordProcessor=None, scoringMethod='euclidean', disp=False):
    """
    Iteratively masks all knowledge tokens in document string and determines
    score relative to initial vector. Returns dict of token and score
        -document: string of the document to vectorize
        -maskToken: string to replace each token with
        -scoringMethod: use euclidean distance or dot product to determine distance from baseVec
        -disp: whether to display the bar chat of distances
    """
    # assertions and special conditions
    assert isinstance(document, str), "document must have type 'str'"
    assert isinstance(maskToken, str), "maskToken must have type 'str'"
    assert (scoringMethod in ['euclidean', 'dot']), "scoringMethod must be 'euclidean' or 'dot'"

    if not keywordProcessor:
        keywordProcessor = build_keyword_processor(document.split())

    # define scoring method
    if (scoringMethod=='euclidean'):
        def calc_score(maskedVec, baseVec):
            return euclidean(maskedVec, baseVec)
    elif (scoringMethod=='dot'):
        def calc_score(maskedVec, baseVec):
            return np.sum(maskedVec * baseVec) / np.linalg.norm(baseVec)

    # calculate vector of raw document
    baseVec = docVecs.vectorize_doc(document)

    # find tokens in document with both greedy and non-greedy matching
    foundTokens = set(keywordProcessor.extract_keywords(document))

    scoreDict = {}

    for token in foundTokens:
        print(colored(f'\t{token}', 'red'), end=' | ')
        maskedDoc = re.sub(token, maskToken, document)
        maskedVec = docVecs.vectorize_doc(maskedDoc)
        score = calc_score(maskedVec, baseVec)
        print(colored(score, 'green'))
        scoreDict.update({token:score})

    if disp:
        plt.bar(scoreDict.keys(), scoreDict.values())
        plt.ylabel('Euclidean Distance from Base Vector')
        plt.title(f'Tokens Iteratively Replaced With "{maskToken}"')
        plt.show()

    return scoreDict
