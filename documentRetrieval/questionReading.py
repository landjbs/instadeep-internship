"""
Module to read natural questions dataset into dataframe storing weighted
map of questions and their vectors to wikipedia articles and their vectors.
"""

import re
import json_lines
import pandas as pd
from time import time
from os import listdir
from bs4 import BeautifulSoup
from termcolor import colored
import matplotlib.pyplot as plt

from utils.cleaner import clean_text, clean_web_text
import vectorizers.docVecs as docVecs

PATH = 'data/inData/natural_questions/v1.0/train'

# matcher for header tags in html text
headerMatcher = re.compile('^h[1-6$]')
# matcher for p tags in html text
pMatcher = re.compile('^P')


def clean_wiki_html(rawHTML):
    """
    Helper to convert raw wiki html to tuple of cleaned title and cleaned text
    """
    # create soup object for the html string
    soupObj = BeautifulSoup(rawHTML, 'html.parser')
    # find title
    title = soupObj.title.text
    # find raw text of page
    rawText = " ".join(re.findall(r'(?<=<P>)[^$]+(?=</P>)',
                                    string=rawHTML,
                                    flags=re.IGNORECASE))
    return clean_text(title), clean_text(rawText)


def read_question_dataset(path, n, outPath=None):
    """
    Reads Google Natural Questions dataset into dataframe
        -path:      The path to the folder containing jsonl files to analyze
        -n:         The number of questions to analyze from each file
        -outPath:   The path to which to save the dataframe
    """
    startTime = time()
    def scrape_wiki_file(file):
        """ Helper pulls information out of wiki file and returns dict """
        if not file.endswith('.jsonl'):
            print(colored(f'WARNING: Cannot analyze "{file}"', 'red'))
        else:
            print(colored(f'Analyzing: "{file}"', 'cyan'))
            fileData = []
            with open(f'{path}/{file}', 'r') as questionFile:
                for i, questionDict in enumerate(json_lines.reader(questionFile)):
                    if i >= n:
                        break
                    print(colored(f'\tReading Questions: {i}', 'yellow'), end='\r')
                    try:
                        answerInfo = questionDict['annotations'][0]
                        shortAnswerInfo = answerInfo['short_answers']
                        longAnswerInfo  = answerInfo['long_answer']

                        pageTokens = questionDict['document_tokens']

                        if not shortAnswerInfo==[]:
                            shortStart = shortAnswerInfo[0]['start_token']
                            shortEnd = shortAnswerInfo[0]['end_token']
                            shortString = " ".join(tokenDict['token']
                                                for tokenDict in pageTokens[shortStart:shortEnd])
                            questionText = questionDict['question_text']
                            # print(f'\n\n{questionText}\n\t{shortString}\n')

                        if not longAnswerInfo==[]:
                            longStart = longAnswerInfo['start_token']
                            longEnd =   longAnswerInfo['end_token']
                            longString = " ".join(tokenDict['token']
                                                for tokenDict in pageTokens[longStart:longEnd])
                            print(f'\n\n{questionText}\n\t{longString}\n')

                        #### VECTORIZATION STUFF #####
                        # get question text and vectorize
                        # questionText = questionDict['question_text']
                        # questionVec = docVecs.vectorize_doc(questionText)
                        #
                        # # get cleaned string of title and text and vectorie
                        # title, text = clean_wiki_html(questionDict['document_html'])
                        # titleVec = docVecs.vectorize_doc(title)
                        # textVec = docVecs.vectorize_doc(text)
                        #
                        # # get list of start locations for each long answer candidate
                        # longAnswerCandidates = questionDict['long_answer_candidates']
                        # longAnswerStarts = [candidate['start_token']
                        #                     for candidate in longAnswerCandidates]
                        #
                        # # get the url of the page
                        # pageUrl = questionDict['document_url']
                        #
                        # # convert question data into dict and append to fileData list
                        # columnDict =  {'pageUrl':           pageUrl,
                        #                 'questionText':     questionText,
                        #                 'questionVec':      questionVec,
                        #                 'titleVec':         titleVec,
                        #                 'textVec':          textVec,
                        #                 'longAnswerStarts': longAnswerStarts}
                        # # yield columnDict
                        # fileData.append(columnDict)

                    except Exception as e:
                        print(colored(f'\tException: {e}', 'red'))

                return fileData

    # scrape files under path and filter out instances of None
    infoList = []
    for file in listdir(path):
        scrapeList = scrape_wiki_file(file)
        if scrapeList:
            infoList += scrapeList

    dataframe = pd.DataFrame(infoList)

    # save to outPath if give
    if outPath:
        dataframe.to_pickle(outPath)

    print(f'{len(infoList)} questions analyzed in {time()-startTime} seconds.')
    return dataframe
