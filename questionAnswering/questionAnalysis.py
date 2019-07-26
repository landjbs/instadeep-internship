from vectorizers.docVecs import vectorize_doc
from functools import reduce
from time import time
from os import listdir
import pandas as pd
from termcolor import colored
import json_lines

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

                        longStart = longAnswerInfo['start_token']
                        longEnd =   longAnswerInfo['end_token']
                        longString = " ".join(tokenDict['token']
                                            for tokenDict in pageTokens[longStart:longEnd])
                        longVec = vectorize_doc(longString)

                        # convert question data into dict and append to fileData list
                        columnDict =  {'questionText':      questionText,
                                        'questionVec':      questionVec,
                                        'longVec':          longVec}
                                        
                        fileData.append(columnDict)

                    except Exception as e:
                        print(colored(f'\tException: {e}', 'red'))

                return fileData

    startTime = time()
    # fold scraper on files under path to create list of columnDicts
    fold_info_list = lambda prev, file : prev + scrape_wiki_file(file)
    infoList = reduce(fold_info_list, listdir(path), [])
    # convert list to dataframe
    dataframe = pd.DataFrame(infoList)

    # save to outPath if give
    if outPath:
        dataframe.to_pickle(outPath)

    print(f'{len(infoList)} questions analyzed in {time()-startTime} seconds.')
    return dataframe
