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

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

class Model:
    models = []

    def __init__(self, name, mean, std):
        """ Model constructor """
        self.name = name
        self.mean = mean
        self.std = std

    @staticmethod
    def getHighestScore():
        """ Get the highest score algorithm method """
        __score = []
        __results = []
        for model in Model.models:
            __score.append(model.mean)
        __score.sort()
        for i in range(len(Model.models)):
            if Model.models[i].mean == __score[len(__score) - 1]:
                __results.append(Model.models[i])
        return __results

    @staticmethod
    def getAlgorithmModel(name):
        """ Generate the result from getHighestScore name attribute """
        return {
            'Logistic Regression' : LogisticRegression(),
            'Linear Discrimination Analysis' : LinearDiscriminantAnalysis(),
            'DecissionTreeClassifier' : DecisionTreeClassifier(),
            'SVM' : SVC(),
            'Gaussian NB' : GaussianNB(),
            'K NeighborsClassifier' : KNeighborsClassifier()
        }.get(name, None)

