import re
from functools import reduce
from time import time
from os import listdir
import pandas as pd
from termcolor import colored
import json_lines
from utils.cleaner import clean_web_text
# from vectorizers.docVecs import vectorize_doc


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
                        # questionVec = vectorize_doc(questionText)

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

                        # longVec = vectorize_doc(longText)

                        colomnDict = {'questionText': questionText,
                                        'questionVec': questionVec}

                        # get vec of all other paragraphs
                        nonAnswerTokens = pageTokens[:longStart] + pageTokens[longEnd:]
                        # compile tokens outside of answer into single string
                        nonAnswerHTML = " ".join(tokenDict['token']
                                            for tokenDict in nonAnswerTokens)

                        paragraphs = re.findall(r'(?<=<P>)[^$]+(?=</P>)',
                                                string=nonAnswerHTML,
                                                flags=re.IGNORECASE)

                        for paragraph in paragraphs:
                            print(clean_web_text(paragraph))

                        # def process_paragraph(paragraph):
                        #     cleanParagraph = clean_web_text(paragraph)
                        #     paraVec = vectorize_doc(cleanParagraph)
                        #     paraLen = len(cleanParagraph.split())
                        #
                        #
                        # paraDict = {f'wrong{i}':process_paragraph(para)
                        #             for para in paragraphs}
                        # print(paraDict)




                        longString = " ".join(tokenDict['token']
                                            for tokenDict in pageTokens[longStart:longEnd])
                        longVec = vectorize_doc(longString)

                        # convert question data into dict and append to fileData list
                        columnDict =  {'questionText':      questionText,
                                        'questionVec':      questionVec,
                                        'longVec':          longVec}
                        columnDict = {}

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
