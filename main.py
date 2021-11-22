# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math

from Model import loadModel, prepareData
from WebScapper import get_reviews
from flask import Flask, jsonify, request
import numpy as np
import pandas as pd


app = Flask(__name__)

app.config["DEBUG"] = True

model = loadModel()

def detectFakeReviews(url = None):

    #this should be loaded when server starts


    #getting all reviews from scrapping.
    data = get_reviews(url)

    #prepare data
    data = prepareData(data)

    #predict on data
    pred = model.predict(data)
    return pred

@app.route('/api/v1/', methods=['POST'])
def index():
    web_address = request.get_json()['web_address']

    data = np.array(detectFakeReviews(url=web_address)).flatten().tolist()
    res = { "data": data }

    return jsonify(res)

app.run()

#
# if __name__ == '__main__':
#     detectFakeReviews(link)

