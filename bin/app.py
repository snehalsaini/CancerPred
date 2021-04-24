import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import build_dataset
import joblib
import pandas as pd
from __init__ import *
from sklearn.ensemble import RandomForestClassifier


app = Flask(__name__)
model = joblib.load("./random_forest.joblib")

def calculate_metrics(true_values, predicted_values):
    """
    Calculate precision, recall, f1-score and support based on classifier output

    :param true_values: actual class output
    :param predicted_values: predicted class output
    :return: calculated metrics
    """

    return metrics.classification_report(y_true=true_values, y_pred=predicted_values)

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
    train, test = build_dataset.main()
    k = random_forest_model(train, test)
    #model = joblib.load("./random_forest.joblib")
    #k = model.predict(X=test[test.columns[:-1]])

    return render_template('index.html', prediction_text='OUTPUT =  {}'.format(k))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)