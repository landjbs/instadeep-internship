"""
Modularized document vectorization tools
"""

import re
from math import floor
import numpy as np
from bert_serving.client import BertClient


# bert-serving-start -model_dir /Users/landonsmith/Desktop/uncased_L-24_H-1024_A-16 -num_worker=1

# bert-serving-start -pooling_strategy NONE -model_dir /Users/landonsmith/Desktop/shortBert -num_worker=1 -mask_cls_sep

bc = BertClient(check_length=False)

sentenceMatcher = re.compile(r'[!|.|?]')


### Vectorization Methods ###
def vectorize_doc(document):
    """ Vectorizes entire document to 1024 dimensions using BertClient """
    return bc.encode([document])[0]


def vectorize_sentence_split(document, sentenceDelimiters=['!', '.', '?']):
    """ Vectorizes document as list of sentences; returns list of vectors """
    # build matcher
    if not (sentenceDelimiters==['!','.','?']):
        sentenceMatcher = re.compile(f"[{'|'.join(delimiter for delimiter in sentenceDelimiters)}]")
    # tokenize sentences and remove empty sentences
    splitDoc = [sentence for sentence in re.split(sentenceMatcher, document)
                if not (sentence=="")]
    return bc.encode(splitDoc)


def vectorize_n_split(document, n):
    """
    Vectorizes document as matrix of vector embeddings of n fractions of the
    document. Chunks are delimited by whitespace by not by sentence endings
    and vary in size by up to n-1. Returned docMatrix will always be (n, 1024)
    """
    # split document into words and find length
    words = document.split()
    numWords = len(words)
    # split words into n roughly-even-sized chunks--method function of (n, numWords)
    if numWords < n:
        raise ValueError(f'Document must have more than {n} words.')
    elif ((numWords % n) == 0):
        chunkSize = int(numWords / n)
        chunkMatrix = [" ".join(words[i:i+chunkSize])
                        for i in range(0, numWords, chunkSize)]
    else:
        # calculate size of first chunk and size of others
        baseChunkSize = floor(len(words) / n)
        firstChunkSize = baseChunkSize + (numWords % n)
        # initialize chunkMatrix and add first chunk
        chunkMatrix = []
        chunkMatrix.append(" ".join(words[0:firstChunkSize]))
        # add remaining chunks of baseChunkSize
        for i in range(firstChunkSize, numWords, baseChunkSize):
            chunkMatrix.append(" ".join(words[i:i+baseChunkSize]))
    # create matrix of vectorized chunks
    docMatrix = np.array([vectorize_doc(chunk) for chunk in chunkMatrix])
    return docMatrix
