import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template

def DNAseq():
    ## to read the dataset
    data = pd.read_table('human_data.txt')

    def getKmers(sequence, size=6):
        return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]
    ##since size=6, obtained kmers would be called hexamers

    data['words'] = data.apply(lambda x: getKmers(x['sequence']), axis=1)
    ## apply kmers function to the sequence coloumn of data to get hexamers

    human_texts = list(data['words'])
    for item in range(len(human_texts)):
        human_texts[item] = ' '.join(human_texts[item])   

    y_data = data.iloc[:, 1].values ##to extract the gene class coloumn that will become our output array

    # The n-gram size of 4 was previously determined by testing
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(ngram_range=(4,4))
    X = cv.fit_transform(human_texts)

    data['class'].value_counts().sort_index().plot.bar()
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y_data, test_size = 0.20, random_state=42)
    from sklearn.naive_bayes import MultinomialNB
    classifier = MultinomialNB(alpha=0.1) ## The alpha parameter was determined by grid search previously
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    return y_pred

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    pred = DNAseq()
    output = pred[:10]
    return render_template('index.html', prediction_text='DNA classes for first 10 sequences in test data {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
