import pandas as pd
from math import inf
from keras.models import load_model
from scipy.spatial.distance import euclidean

from vectorizers.docVecs import vectorize_doc
import vectorizers.datasetVectorizer as datasetVectorizer

class Retreiver():
    """
    Class to store document vectors and retreive question answers with a
    single call
    """
    def __init__(self, documentsCsv, n, sep=','):
        """
        Initializes retreiver object with vectorized files
        """
        documentsDict = {}
        with open(documentsCsv, 'r') as documentsFile:
            for i, line in enumerate(documentsFile):
                if i > n:
                    break
                print(f'Reading File: {i}')
                sepLoc = line.find(sep)
                title = line[:sepLoc]
                text = line[sepLoc:]
                textVec = vectorize_doc(text)
                documentsDict.update({title:textVec})
        self.documents = documentsDict
        self.model = load_model('data/outData/models/documentRetrievalModel.sav')

    def display_titles(self):
        """ Displays all the titles stored in the retriver object """
        for title in (self.documents).keys():
            print(title)

    def retrieve(self, question, n, cutoff=None):
        """ Retrieves top n files stored in retriver object """
        if not cutoff:
            cutoff = inf

        questionVec = vectorize_doc(question)

        scoresList = []

        for docTitle, docVec in (self.documents).iteritems():
            docDist = euclidean(questionVec, docVec)
            if docDist < cutoff:
                distDict = vec_to_dict(docDist)
                distDf = pd.DataFrame(distDict)
                prediction = self.model.predict(distDf)
                print(f'{docTitle}: {prediction}')
                scoresList.append((prediction, docTitle))
        scoresList.sort(reverse=True)
        return [scoreTuple[1] for scoreTuple in scoresList[:n]]
