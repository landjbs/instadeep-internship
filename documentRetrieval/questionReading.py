"""
Module to read natural questions dataset into dataframe storing weighted
map of questions and their vectors to wikipedia articles and their vectors.
"""

import re
import json_lines
import pandas as pd
from os import listdir
from bs4 import BeautifulSoup

from utils.cleaner import clean_text, clean_web_text
import vectorizers.docVecs as docVecs

PATH = 'data/inData/natural_questions/v1.0/train'

# matcher for header tags in html text
headerMatcher = re.compile('^h[1-6$]')
# matcher for p tags in html text
pMatcher = re.compile('^p')


def analyze_wiki_html(rawHTML):
    """ Helper to convert raw wiki html to a clean div dict """
    # create soup object for the html string
    soupObj = BeautifulSoup(rawHTML, 'html.parser')
    # find title
    title = soupObj.title.text
    # find raw text of page
    rawText = " ".join(str(p) for p in soupObj.find_all(pMatcher, text=True))
    return {'title':title,
                'text': clean_web_text(rawText)}


def read_question_dataset(path):
    """
    Reads Google Natural Questions dataset into dataframe
        -path: the path to the folder containing jsonl files to analyze
    """
    files = listdir(path)
    for file in files:
        if file.endswith('.jsonl'):
            with open(f'{path}/{file}', 'r') as questionFile:
                for i, questionDict in enumerate(json_lines.reader(questionFile)):
                    # get question text and vectorize
                    questionText = questionDict['question_text']
                    questionVec = docVecs.vectorize_doc(questionText)
                    # get dict of page divisions and vectorize
                    divDict = analyze_wiki_html(questionDict['document_html'])
                    longAnswerCandidates = questionDict['long_answer_candidates']
                    longAnswerStarts = [candidate['start_token']
                                        for candidate in longAnswerCandidates]
