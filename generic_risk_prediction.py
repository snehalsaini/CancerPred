import build_dataset
from classifiers import logistic_model, random_forest_model, naive_bayes_model
from __init__ import *
import joblib 


def main():
    """
    Execute generic classification methods on DNA methylation data

    :return: metrics of each classifier
    """

    train, test = build_dataset.main()
    #train, test = build_dataset.normalize_datasets(train, test)

    #logger.info('Applying Logistic Regression')
    #logistic_model(train, test)

    logger.info('Applying Random Forest')
    random_forest_model(train, test)

    #logger.info('Applying Gaussian Naive Bayes')
    #naive_bayes_model(train, test)

    
    
    # Saving model to disk
    #pickle.dump(regressor, open('model.pkl','wb'))

    # Loading model to compare the results
    #model = pickle.load(open('model.pkl','rb'))


if __name__ == '__main__':
    main()
