from vectorizers.docVecs import vectorize_doc


def build_question_database(path, n, outPath=None):
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
                        # shortAnswerInfo = answerInfo['short_answers']

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
                        longVec = vectorize_doc


                        # get the url of the page
                        pageUrl = questionDict['document_url']

                        # convert question data into dict and append to fileData list
                        columnDict =  {'pageUrl':           pageUrl,
                                        'questionText':     questionText,
                                        'questionVec':      questionVec,
                                        'titleVec':         titleVec,
                                        'textVec':          textVec,
                                        'longAnswerStarts': longAnswerStarts}
                        # yield columnDict
                        fileData.append(columnDict)

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
