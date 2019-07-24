import pandas as pd
from keras.models import load_model
import vectorizers.docVecs as docVecs
from vectorizers.datasetVectorizer import vec_to_dict

sentimentModel = load_model('data/outData/models/folderModel5000.sav')

def analyze_text(text):
    textVec = docVecs.vectorize_doc(text)
    vecDict = [vec_to_dict(textVec)]
    df = pd.DataFrame(vecDict)
    prediction = sentimentModel.predict(df)
    print(prediction)
