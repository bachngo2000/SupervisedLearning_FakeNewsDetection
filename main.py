import numpy as np
import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
from math import sqrt

def stemming(data):
    port_stem = PorterStemmer()
    stemmed_data = re.sub('[^a-zA-Z]', ' ', data)
    stemmed_data = stemmed_data.lower()
    stemmed_data = stemmed_data.split()
    stemmed_data = [port_stem.stem(word) for word in stemmed_data if not word in stopwords.words('english')]
    stemmed_data = ' '.join(stemmed_data)
    return stemmed_data

def calc_euclidean_distance(coordinate1, coordinate2):
    return np.sqrt(np.sum((coordinate1 - coordinate2).power(2)))

class K_Nearest_Neighbors:
    def __init__(self, k=100):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [calc_euclidean_distance(x, x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_idx = np.argsort(distances)[: self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_neighbor_labels = [self.y_train[i] for i in k_idx]
        # return the most common class label
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]


if __name__ == '__main__':

    # stopwords in English
    print(stopwords.words('english'))

    # Data Pre-processing:
    # loading the dataset to a pandas DataFrame
    news_dataset = pd.read_csv('news_data.csv')

    # print the first 5 rows of the dataFrame
    print(news_dataset.head())

    # counting the number of missing values in the dataset
    print(news_dataset.isnull().sum())

    # replacing the null values with empty string
    news_dataset = news_dataset.fillna('')

    # separating the data and label
    X = news_dataset.drop(columns='eval', axis=1)
    Y = news_dataset['eval']

    print(X)
    print(Y)

    # Stemming: process of reducing a word to its Root word
    # Ex: actor, actress, acting --> act
    news_dataset['title'] = news_dataset['title'].apply(stemming)
    print(news_dataset['title'])

    # separating the data and eval
    X = news_dataset['title'].values
    Y = news_dataset['eval'].values
    print(X)
    print(Y)

    # Convert the textual data to numerical data
    vectorizer = TfidfVectorizer()
    vectorizer.fit(X)

    X = vectorizer.transform(X)

    print("This is X: \n", X)
    print("This is Y: \n", Y)


    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

    def accuracy(y_true, y_pred):
        acc = np.sum(y_true == y_pred) / len(y_true)
        return acc

    print("Size of X_train: \n", )
    ##
    k = 13
    clf = K_Nearest_Neighbors(k=k)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    print("KNN classification accuracy", accuracy(y_test, predictions))
