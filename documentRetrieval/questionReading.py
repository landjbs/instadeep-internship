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
pMatcher = re.compile('^P')


def clean_wiki_html(rawHTML):
    """ Helper to convert raw wiki html to a clean div dict """
    # create soup object for the html string
    soupObj = BeautifulSoup(rawHTML, 'html.parser')
    # find title
    title = soupObj.title.text
    # find raw text of page
    # rawText = " ".join(str(p) for p in soupObj.find_all(pMatcher))
    rawText = " ".join(re.findall(r'(?<=<P>)[^</P>]+(?=</P>)',
                                    string=rawHTML,
                                    flags=re.IGNORECASE))
    return {'title':title,
                'text': clean_web_text(rawText)}


def read_question_dataset(path):
    """
    Reads Google Natural Questions dataset into dataframe
        -path: the path to the folder containing jsonl files to analyze
    """

    def scrape_wiki_file(file):
        """ Helper pulls information out of wiki file and returns dict """
        if file.endswith('.jsonl'):
            with open(f'{path}/{file}', 'r') as questionFile:
                for i, questionDict in enumerate(json_lines.reader(questionFile)):
                    # get question text and vectorize
                    questionText = questionDict['question_text']
                    # questionVec = docVecs.vectorize_doc(questionText)

                    # get dict of page divisions and vectorize
                    divDict = clean_wiki_html(questionDict['document_html'])
                    # divVecs = {divName:(docVecs.vectorize_doc(divText))
                    #             for divName, divText in divDict.items()}

                    # get list of start locations for each long answer candidate
                    longAnswerCandidates = questionDict['long_answer_candidates']
                    longAnswerStarts = [candidate['start_token']
                                        for candidate in longAnswerCandidates]
                    if divDict['text']=="": print(questionDict['document_html'])

    # scrape files in
    infoList = [scrape_wiki_file(file) for file in listdir(path)]
    dataframe = pd.DataFrame(infoList)
    return dataframe
