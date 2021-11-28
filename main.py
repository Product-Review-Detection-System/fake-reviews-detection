# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math

from Model import loadModel, prepareData
from WebScapper import get_reviews
from flask import Flask, jsonify, request
import numpy as np


app = Flask(__name__)

app.config["DEBUG"] = True

model = loadModel()

def detectFakeReviews(url = None):

    #this should be loaded when server starts


    #getting all reviews from scrapping.
    rawData = get_reviews(url)

    #prepare data
    reviewData = prepareData(rawData)

    #predict on data
    pred = model.predict(reviewData['padData'])
    return {
        "predictions": pred,
        "reviews": reviewData['reviews'],
        "ratings": rawData['model']['ratings'],
        "itemId": rawData["model"]['item']["itemId"],
        "itemTitle": rawData["model"]['item']["itemTitle"],
        "itemPic": rawData["model"]['item']["itemPic"],
        "itemUrl": rawData["model"]['item']["itemUrl"]
    }

@app.route('/api/v1/', methods=['POST'])
def index():
    web_address = request.get_json()['web_address']

    result = detectFakeReviews(url=web_address)
    result['predictions'] = np.array(result['predictions']).flatten().tolist()
    result['reviews'] = np.array(result['reviews']).flatten().tolist()
    res = {"result": result}

    return jsonify(res)



if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0")

