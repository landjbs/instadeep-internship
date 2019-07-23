import re
import numpy as np
from scipy.spatial.distance import euclidean

import docVecs

binopList = ['+', '-', '?', '==']
unopList = ['(', ')']
unopMatcher = re.compile(r'[(|)]')


def bert_parser(inStr):
    tokens = inStr.split()
    tokenNum = len(tokens)
    if (tokenNum==0):
        raise ValueError("Cannot parse empty string.")
    elif (tokenNum==1):
        cleanToken = re.sub(unopMatcher, '', tokens[0])
        return docVecs.vectorize_doc(cleanToken)
    else:
        evaluateTokens = re.findall(r'(?<=\()[^)]+(?=\))', inStr)
        vectorizedTokens = {token:docVecs.vectorize_doc(token) for token in evaluateTokens}
        for i, token in enumerate(tokens):
            cleanToken = re.sub(unopMatcher, '', token)
            if cleanToken in vectorizedTokens:
                tokens[i] = vectorizedTokens[cleanToken]
        for i in range(0, len(tokens), 3):
            token1 = tokens[i]
            token2 = tokens[i+2]
            operator = tokens[i+1]
            print(token1, token2, operator)
            print(bert_binop(token1, token2, operator))


def find_subs(inStr):
    numOpen = 0
    subsList = []
    readText = ""
    for c in inStr:
        if c == '(':
            numOpen += 1
        elif c == ')':
            numOpen -= 1
            if numOpen == 0:
                subsList.append(readText)
                readText = ""
        if numOpen == 0:
            pass
        else:
            readText += c
    return subsList


def bert_multiParser(inStr):
    expression = re.findall(r'(?<=\().+(?=\))', inStr)
    subExpressions = find_subs(expression[0])
    if len(subExpressions)==1:
        return docVecs.vectorize_doc(subExpressions[0])
    elif len(subExpressions)==2:
        left = bert_multiParser(subExpressions[0])
        right = bert_multiParser(subExpressions[1])
        leftEnd = (expression[0].find(subExpressions[0])) + len(subExpressions[0])
        rightStart = expression[0].find(subExpressions[1])
        operator = re.findall(r'[?|==|\-|\+]', expression[0][leftEnd:rightStart])
        return bert_binop(left, right, operator[0])


def bert_binop(vec1, vec2, operator):
    """ Performs binary operation from binopList on vec1 on vec2 """
    assert (len(vec1)==len(vec2)==1024), "Vectors must both have length of 1024"
    if (operator=='+'):
        return np.add(vec1, vec2)
    elif (operator=='-'):
        return np.subtract(vec1, vec2)
    elif (operator=='?'):
        return euclidean(vec1, vec2)
    elif (operator=='=='):
        return (vec1 == vec2)
    else:
        raise ValueError(f"Invalid operator '{operator}'")


def bert_arthimetic(inStr):
    """ inStr must have form 'TERM_1 [+|-] TERM_2 '"""
    splitStrs = inStr.split()
    numTokens = len(splitStrs)

    if (numTokens==1):
        return docVecs.vectorize_doc(splitStrs)

    elif (numTokens==3):
        operator = splitStrs[1]
        vecs = docVecs.vectorize_doc_list([splitStrs[0], splitStrs[2]])
        # identify arthimetic method
        if (operator=='+'):
            return np.add(vecs[0], vecs[1])
        elif (operator=='-'):
            return np.subtract(vecs[0], vecs[1])
        else:
            raise ValueError('Invalid operator')
