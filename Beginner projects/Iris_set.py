# make predictions
import pickle
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)
print('Describe dataset')
print(dataset.describe(),'\n')
print('Group by class')
print(dataset.groupby('class').size(),'\n')

#Split dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.20, random_state=1)

#Make validations on whole dataset
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

#Evaluate predictions
print('Evaluate predictions')
print('Accuracy score')
print(accuracy_score(Y_validation, predictions),'\n')
print('Confusion matrix')
print(confusion_matrix(Y_validation, predictions))
print('Classification report')
print(classification_report(Y_validation, predictions))

#Save the model for future use
#filename = 'Iris-model.sav'
#pickle.dump(model,open(filename,'wb'))