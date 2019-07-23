import pandas as pd
from os import listdir

import docVecs
from cleaner import clean_text

def vec_to_dict(docVec):
    """ Converts docVec to dict mapping dimension names to values """
    return {dimension:value for dimension, value in enumerate(docVec)}


def vectorize_folderPath(folderPath, cleanFiles=True, outPath=""):
    """
    Vectorizes each file under folderPath and returns dataframe
        -folderPath: the path to the folder in which the files are stored
        -cleanFiles: true if the file text should be cleaned before vectorization
        -outPath: location at which to save the dataframe
    """

    def build_file_dict(file):
        """ Helper to build a dict storing file vector and name """
        with open(f'{folderPath}/{file}', 'r') as fileObj:
            # clean the text before vectorizing if cleanFiles
            text = clean_text(fileObj.read()) if cleanFiles else fileObj.read()
            # vectorize text and convert vector to dict
            fileVector = docVecs.vectorize_doc(text)
            fileDict = vec_to_dict(fileVector)
            # add filename to the dict and return
            fileDict.update({'fileName':file})
            return fileDict

    # build list of fileDicts from files under folderPath
    fileList = [build_file_dict(file) for file in listdir(folderPath)]
    # convert list to dataframe
    dataframe = pd.DataFrame(fileList)
    # save dataframe if prompted
    if not (outPath==""):
        dataframe.to_pickle(outPath)
    return dataframe


def vectorize_csv(filePath, delimiter=',', cleanFiles=True, outPath=""):
    """
    Builds a dataframe of vectors from csv where lines have form:
    'title DELIMITER text'.
        -filePath: path the the csv file to analyze
        -delimited: the delimiter used to separate title and text
        -cleanFiles: true if the file text should be cleaned before vectorization
        -outPath: location at which to save the dataframe
    """
    fileList = []
    with open(filePath, 'r') as csvFile:
        for i, line in enumerate(csvFile):
            # only the first instace of the delimiter is used to split
            delimLoc = line.find(delimiter)
            title, rawText = line[:delimLoc], line[delimLoc:]
            cleanText = clean_text(rawText) if cleanFiles else rawText
            textVector = docVecs.vectorize_doc(cleanText)
            textDict = vec_to_dict(textVector)
            textDict.update({'fileName':title})
            fileList.append(textDict)
            print(f'Building dataframe: {i}', end='\r')
    # build dataframe from text vectors
    dataframe = pd.DataFrame(fileList)
    if not (outPath==""):
        dataframe.to_pickle(outPath)
    return dataframe
