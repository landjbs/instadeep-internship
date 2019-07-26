import re
from functools import reduce
from time import time
from os import listdir
import pandas as pd
from termcolor import colored
import json_lines
from utils.cleaner import clean_web_text
from vectorizers.docVecs import vectorize_doc

def build_question_database(path, n, outPath=None):
    """
    Reads Google Natural Questions dataset into dataframe
        -path:      The path to the folder containing jsonl files to analyze
        -n:         The number of questions to analyze from each file
        -outPath:   The path to which to save the dataframe
    """

    def scrape_wiki_file(file):
        """ Helper pulls information out of wiki file and returns dict """
        if not file.endswith('.jsonl'):
            return []
        else:
            print(colored(f'Analyzing: "{file}"', 'cyan'))
            fileData = []
            with open(f'{path}/{file}', 'r') as questionFile:
                for i, questionDict in enumerate(json_lines.reader(questionFile)):
                    if i >= n:
                        break
                    print(colored(f'\tReading Questions: {i}', 'yellow'), end='\r')
                    try:
                        # get question text and vectorize
                        questionText = questionDict['question_text']
                        questionVec = vectorize_doc(questionText)

                        # get list of start locations for each long answer candidate
                        answerInfo = questionDict['annotations'][0]
                        longAnswerInfo  = answerInfo['long_answer']
                        pageTokens = questionDict['document_tokens']

                        if longAnswerInfo==[]:
                            raise ValueError("No long answer text.")

                        # get attributes of long answer
                        longStart = longAnswerInfo['start_token']
                        longEnd =   longAnswerInfo['end_token']
                        longText = " ".join(tokenDict['token']
                                        for tokenDict in pageTokens[longStart:longEnd])

                        # clean and vectorize long answer text
                        longVec = vectorize_doc(clean_web_text(longText))

                        columnDict = {'questionText':   questionText,
                                        'questionVec':  questionVec,
                                        'paraVec':      longVec,
                                        'score':        1}

                        fileData.append(columnDict)

                        # get vec of all other paragraphs
                        nonAnswerTokens = pageTokens[:longStart] + pageTokens[longEnd:]
                        # compile tokens outside of answer into single string
                        nonAnswerHTML = " ".join(tokenDict['token']
                                            for tokenDict in nonAnswerTokens)
                        # get list of nonAnswer paragraphs
                        paragraphs = re.findall(r'(?<=<P>)[^$]+(?=</P>)',
                                                string=nonAnswerHTML,
                                                flags=re.IGNORECASE)

                        for paragraph in paragraphs:
                            curColumnDict = columnDict.copy()
                            paraVec = vectorize_doc(clean_web_text(paragraph))
                            curColumnDict.update({'paraVec': paraVec,
                                                    'score': 0})
                            fileData.append(curColumnDict)

                    except Exception as e:
                        print(colored(f'\tException: {e}', 'red'))

                return fileData

    startTime = time()
    # fold scraper on files under path to create list of columnDicts
    fold_info_list = lambda prev, file : prev + scrape_wiki_file(file)
    infoList = reduce(fold_info_list, listdir(path), [])
    # convert list to dataframe
    dataframe = pd.DataFrame(infoList)
    print(dataframe)
    # save to outPath if give
    if outPath:
        dataframe.to_pickle(outPath)

    print(f'{len(infoList)} questions analyzed in {time()-startTime} seconds.')
    return dataframe
