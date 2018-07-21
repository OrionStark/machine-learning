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
from model import Model

class ModelsEvaluation:
    __models = []
    def __init__(self, x_train, y_train):

        """ Models Evaluation Constructor
        Initiallize 6 models """

        self.x_train = x_train
        self.y_train = y_train

        # 6 default algorith models
        # Maybe I will add more algorithm in the future
        self.__models.append(('Logistic Regression', LogisticRegression()))
        self.__models.append(('Linear Discrimination Analysis', LinearDiscriminantAnalysis()))
        self.__models.append(('DecissionTreeClassifier', DecisionTreeClassifier()))
        self.__models.append(('SVM', SVC()))
        self.__models.append(('Gaussian NB', GaussianNB()))
        self.__models.append(('K NeighborsClassifier', KNeighborsClassifier()))

    # Evaluate the accuracy of 6 models
    # K fold Validation model
    def evaluateAccuracy(self):
        Model.models[:] = []

        """ Evalueate the accuracy model by given data train. 
            So it could get the best Algorithm to use """

        __results = []
        __names = []
        for name, model in self.__models:
            kfold = model_selection.KFold(n_splits = 10, random_state = 7)
            cv_results = model_selection.cross_val_score(model, self.x_train, self.y_train, cv = kfold, scoring = 'accuracy')
            __results.append(cv_results)
            __names.append(name)
            Model.models.append(Model(name, cv_results.mean(), cv_results.std()))

        # NOT IMPLEMENTED YET
        # 
        # if ( len(Model.getHighestScore()) > 1 ):
        #     if (Model.getHighestScore()[0].mean == Model.getHighestScore()[1].mean):
        #         return Model.getHighestScore()[0]
        #     else:
        #         return Model.getHighestScore()
        # else:
        #     return Model.getHighestScore()[0]

        return Model.getHighestScore()[0]
    
    # LOOCV Validation Model
    def leaveOneOutCrossValidationEvaluation(self):
        Model.models[:] = []

        __results = []
        __names = []
        __looCrossValidation = model_selection.LeaveOneOut()
        for name, model in self.__models:
            cv_results = model_selection.cross_val_score(model, self.x_train, self.y_train, cv = __looCrossValidation)
            __results.append(cv_results)
            __names.append(name)
            Model.models.append(Model(name, cv_results.mean(), cv_results.std()))

        return Model.getHighestScore()[0]

    
