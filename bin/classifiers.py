import pandas as pd
from __init__ import *
from sklearn import metrics, linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


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
    logger.info(predicted[:10])

    logger.info('Calculated metrics')
    logger.info(calculate_metrics(test['Tumor'], predicted) + '\n')


    #joblib.dump(forest_model, "./random_forest.joblib")
    
