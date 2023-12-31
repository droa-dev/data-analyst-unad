import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

ds = pd.read_csv("Cleaned-Data.csv")
ds.info()

ds.isnull().sum()

feature_cols = ['Fever', 'Tiredness', 'Dry-Cough', 'Difficulty-in-Breathing',
       'Sore-Throat', 'None_Sympton', 'Pains', 'Nasal-Congestion',
       'Runny-Nose', 'Diarrhea', 'None_Experiencing']
#objetive_cols = ['Severity_None', 'Severity_Mild', 'Severity_Moderate', 'Severity_Severe']
objetive_col1 = 'Severity_None'
objetive_col2 = 'Severity_Mild'
objetive_col3 = 'Severity_Moderate'
objetive_col4 = 'Severity_Severe'

X = ds[feature_cols]
y1 = ds[objetive_col1]
y2 = ds[objetive_col2]
y3 = ds[objetive_col3]
y4 = ds[objetive_col4]

X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y1, test_size = 0.3, random_state=0)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y2, test_size = 0.3, random_state=0)
X_train3, X_test3, y_train3, y_test3 = train_test_split(X, y3, test_size = 0.3, random_state=0)
X_train4, X_test4, y_train4, y_test4 = train_test_split(X, y4, test_size = 0.3, random_state=0)

def perform(y_test, y_pred, name_cols):
    print("Precision : ", precision_score(y_test, y_pred))
    print("Recall : ", recall_score(y_test, y_pred))
    print("Accuracy : ", accuracy_score(y_test, y_pred))
    print("F1 Score : ", f1_score(y_test, y_pred))
    print('')
    print(confusion_matrix(y_test, y_pred), '\n')
    cm = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred))
    cm.plot()

knn = KNeighborsClassifier(n_neighbors = 10) #setting up the KNN model to use 5NN
knn.fit(X_train1, y_train1) #fitting the KNN

#Checking performance on the training set
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train1, y_train1)))
#Checking performance on the test set
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test1, y_test1)))

y_pred_knn = knn.predict(X_test1)
perform(y_test1, y_pred_knn, objetive_col1)
print(classification_report(y_test1, y_pred_knn))

knn.fit(X_train2, y_train2) #fitting the KNN

#Checking performance on the training set
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train2, y_train2)))
#Checking performance on the test set
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test2, y_test2)))

y_pred_knn = knn.predict(X_test2)
perform(y_test2, y_pred_knn, objetive_col2)
print(classification_report(y_test2, y_pred_knn))

knn.fit(X_train3, y_train3) #fitting the KNN

#Checking performance on the training set
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train3, y_train3)))
#Checking performance on the test set
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test3, y_test3)))

y_pred_knn = knn.predict(X_test3)
perform(y_test3, y_pred_knn, objetive_col3)
print(classification_report(y_test3, y_pred_knn))

knn.fit(X_train4, y_train4) #fitting the KNN

#Checking performance on the training set
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train4, y_train4)))
#Checking performance on the test set
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test4, y_test4)))

y_pred_knn = knn.predict(X_test4)
perform(y_test4, y_pred_knn, objetive_col4)
print(classification_report(y_test4, y_pred_knn))

#scaler
scaler = MinMaxScaler()
ds[feature_cols] = scaler.fit_transform(ds[feature_cols])

ds1 = ds.drop(['Contact_Dont-Know', 'Contact_No', 'Contact_Yes', 'Country', 'None_Experiencing', 'Age_0-9', 'Age_10-19', 'Age_20-24', 'Age_25-59', 'Age_60+', 'Gender_Female', 'Gender_Male', 'Gender_Transgender', 'None_Sympton','Severity_Mild', 'Severity_Moderate', 'Severity_Severe'], axis = 1, inplace = False)
ds.info()
train, test = train_test_split(ds, test_size = 0.3, random_state=0)

X_train = train.iloc[:, :19].values
X_test = test.iloc[:, :19].values
y_train = train.iloc[:, -1].values
y_test = test.iloc[:, -1].values

param_grid_knn = {
    'n_neighbors': [2, 5, 10, 15],                                   
    'algorithm': ['ball_tree', 'kd_tree', 'brute', 'auto'],          
    'metric': ['minkowski', 'euclidean', 'manhattan', 'chebyshev']
}

kNNModel_grid = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid_knn, verbose=1, cv=10, n_jobs=-1)

kNNModel_grid.fit(X_train, y_train)
print(kNNModel_grid.best_estimator_)

knn = KNeighborsClassifier(algorithm='ball_tree', n_neighbors=2) #setting up the KNN model to use 2NN
knn.fit(X_train, y_train) #fitting the KNN

#Checking performance on the training set
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train, y_train)))
#Checking performance on the test set
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test, y_test)))

y_pred_knn = knn.predict(X_test)
print(classification_report(y_test, y_pred_knn))
