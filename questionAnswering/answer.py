""" Given a text and a question finds the window that best matches the question """

import re
from scipy.spatial.distance import euclidean, cosine
from vectorizers.docVecs import vectorize_doc

def answer_question(question, text):
    questionVec = vectorize_doc(question)

    sentences = re.split(r'[.|!|\?]', text)

    bestDist, bestSentence = 20, "NONE"
    for sentence in sentences:
        if not sentence=="":
            sentenceVec = vectorize_doc(sentence)
            sentenceDist = cosine(questionVec, sentenceVec)
            if sentenceDist < bestDist:
                bestDist = sentenceDist
                bestSentence = sentence

    return bestSentence
