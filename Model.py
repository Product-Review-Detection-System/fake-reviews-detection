import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential, load_model


def loadModel():
    model =  load_model('Word2Vec.hdf5')
    return model

def prepareData(inputData):
    SEQUENCE_LENGTH = 180

    #get the data in form and extract items array from it
    rawData = inputData['model']['items']

    #Extract all the review from each item
    rawData = map(lambda item: item['reviewContent'], rawData)
    data = list(rawData)

    #convert it to dataframe and remove None values
    data = pd.DataFrame(data)
    data = data.dropna().applymap(str).to_numpy()


    #Tokenize data
    # data = np.array(inputData)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data.flatten())
    padData = pad_sequences(tokenizer.texts_to_sequences((data.flatten())), maxlen=SEQUENCE_LENGTH)
    return {"padData": padData, "reviews": data}
