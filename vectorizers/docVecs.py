"""
Modularized document vectorization tools
"""

import re
import numpy as np
from math import floor
from functools import reduce
from bert_serving.client import BertClient


# bert-serving-start -model_dir /Users/landonsmith/Desktop/uncased_L-24_H-1024_A-16 -num_worker=1

# bert-serving-start -pooling_strategy NONE -model_dir /Users/landonsmith/Desktop/shortBert -num_worker=1 -mask_cls_sep -max_seq_len=40

bc = BertClient(check_length=True)


class VectorizationError(Exception):
    """ Class for errors during vectorization """
    pass

sentenceMatcher = re.compile(r'(?<=[\.\!\?;])[^a-zA-Z0-9]')

### Vectorization Methods ###
def vectorize_doc(document):
    """ Vectorizes entire document to 1024 dimensions using BertClient """
    return bc.encode([document])[0]


def get_word_encodings(text, maxLen=35):
    """
    Returns ordered list of word vec embeddings in context of their sentence.
    Can handle text of any length, but raises Exception if any of the sentences
    is longer than max_seq_len of BERT Client.
    Does not include CLS or SEP tokens but does include individual punctuation
    vectors.
    """
    # split by sentence and assert length
    sentences = re.split(sentenceMatcher, text)
    if any(len(sentence.split())>maxLen for sentence in sentences):
        raise VectorizationError(f'Text contains sentence over {maxLen} words.')
    # encode sentences
    sentenceVecs = bc.encode(sentences)
    # build and return list of contextual word embeddings
    textVecs = []
    for wordVecs in sentenceVecs:
        for vec in wordVecs:
            if not (vec[0]==0):
                textVecs.append(vec)
    return textVecs


def vectorize_sentence_split(document, sentenceDelimiters=['!', '.', '?']):
    """ Vectorizes document as list of sentences; returns list of vectors """
    # build matcher
    if not (sentenceDelimiters==['!','.','?']):
        sentenceMatcher = re.compile(f"[{'|'.join(delimiter for delimiter in sentenceDelimiters)}]")
    else:
        sentenceMatcher = re.compile(r'[!|.|?]')
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
