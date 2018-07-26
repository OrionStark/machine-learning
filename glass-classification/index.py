# MIT License

# Copyright (c) 2018 Robby Muhammad Nst

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pandas
import numpy
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from utils.models_evaluation import ModelsEvaluation
from utils.model import Model

def getGlassesClassText(number):
    return {
        1.0 : 'Building windows float processed',
        2.0 : 'Building windows non-float processed',
        3.0 : 'Vehicle windows float processed',
        4.0 : 'Vehicle windows non-float processed',
        5.0 : 'Containers',
        6.0 : 'Tableware',
        7.0 : 'Headlamps'
    }.get(number, None)

# Load the dataset
path = "../datasets/Glass Dataset/glass.data"
names = ['id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'class']
dataset = pandas.read_csv(path, names = names)
array = dataset.values
X = array[:, 1:10]
Y = array[:, 10]
# validation_size = 0.20
# seed = 7
# X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size = validation_size, random_state = seed)
MODEL_EVALUATION = ModelsEvaluation(X, Y)
HIGHEST_ACC_MODEL_NAME = MODEL_EVALUATION.evaluateAccuracy().name
HIGHEST_ACC_MODEL_SCORE = MODEL_EVALUATION.evaluateAccuracy().mean
HIGHEST_MODEL = Model.getAlgorithmModel(HIGHEST_ACC_MODEL_NAME)
HIGHEST_MODEL.fit(X, Y)

predict_data = [
    [ 1.51755, 13.00, 3.60, 1.36, 72.99, 0.57, 8.40, 0.00, 0.11 ],
    [ 1.51574, 14.86, 3.67, 1.74, 71.87, 0.16, 7.36, 0.00, 0.12 ],
    [ 1.51593, 13.09, 3.59, 1.52, 73.10, 0.67, 7.83, 0.00, 0.00 ]
]
if ( HIGHEST_MODEL is not None ):
    prediction = HIGHEST_MODEL.predict(predict_data)
    print("Matching algorithm is " + HIGHEST_ACC_MODEL_NAME)
    for i in range(0, len(prediction)):
        print(getGlassesClassText(prediction[i]))
else:
    print("Didn't get matching algorithm")
