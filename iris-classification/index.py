import pandas
import numpy
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import sys
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
sys.path.append('..')
from utils.models_evaluation import ModelsEvaluation
from utils.model import Model

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size = validation_size, random_state = seed)

MODEL_EVALUATION = ModelsEvaluation(X, Y)
HIGHEST_ACC_MODEL = MODEL_EVALUATION.evaluateAccuracy().name
HIGHEST_ACC_MODEL_SCORE = MODEL_EVALUATION.evaluateAccuracy().mean
HIGHEST_MODEL_INSTANCE = Model.getAlgorithmModel(HIGHEST_ACC_MODEL)
HIGHEST_MODEL_INSTANCE.fit(X, Y)

# Predict data included sepal-length(cm), sepal-width(cm), petal-length(cm), petal-width(cm)
predict_data = [
    [5.7, 2.8, 4.1, 1.3],
    [5.8, 2.7, 5.1, 1.9],
    [7.7, 3.0, 6.1, 2.3]
]
prediction = HIGHEST_MODEL_INSTANCE.predict(predict_data)

if ( prediction is not None ):
    print("Match Algorithm is " + HIGHEST_ACC_MODEL)
    print("Confident : ", HIGHEST_ACC_MODEL_SCORE * 100)
    print(prediction)
    dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
    scatter_matrix(dataset)
    plt.show()
else:
    print("Couldn't find the match algorithm")