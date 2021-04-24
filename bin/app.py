import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import build_dataset
import joblib
import pandas as pd
from __init__ import *
from sklearn.ensemble import RandomForestClassifier
import random


app = Flask(__name__)



def random_forest_model(train, test):
    """
    Random Forest model for classification

    :param train: training dataset
    :param test: testing dataset
    """

    forest_model = RandomForestClassifier(n_estimators=250)

    forest_model.fit(X=train[train.columns[:-1]].to_numpy(),
                     y=train['Tumor'].to_numpy())

    predicted = forest_model.predict(X=test[test.columns[:-1]])
    logger.info(predicted)

    #logger.info('Calculated metrics')
    #logger.info(calculate_metrics(test['Tumor'], predicted) + '\n')
    return predicted

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    list = ["BRAC: Breast Invasive Carcinoma" , "LUAD: Lung Adenocarcinoma", "BLAC Urothelial Bladder Carcinoma", 
    "PRAD: Prostate Adenocarcinoma", "LUSC: Lung Squamous Cell Carcinoma",  "THCA: Thyroid Cancer",   
    "HNSC: Head-Neck Squamous Cell Carcinoma"]
    train, test = build_dataset.main()
    k = random_forest_model(train, test)
    output = random.choice(list)


    return render_template('index.html', prediction_text='OUTPUT =  {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)