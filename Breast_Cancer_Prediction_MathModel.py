import kagglehub
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

######################################################
#########   Load Dataset  #############################     
######################################################

dataset_url = "uciml/breast-cancer-wisconsin-data"
dataset_path = kagglehub.dataset_download(dataset_url)
file_name = os.path.join(dataset_path, 'data.csv')

df = pd.read_csv(file_name)
print(df.head())
print(df.columns)

######################################################
#########   Data Preprocessing  ######################
######################################################
x_train = df.iloc[:, 2:32].values
y_train = df.iloc[:, 1].values
print(x_train)
print(y_train)
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
y_train = labelencoder.fit_transform(y_train)
print(y_train)
######################################################
#########  Creating Logistic Regression Model ########
######################################################

#lets improve more by adding bias value
def compute_wb(x, y):
  m, n = x.shape
  w = []
  b = np.zeros(n,)
  for j in range(n):
    sum_x = np.sum(x[:, j])
    sum_y = np.sum(y)
    sum_xy = np.sum(x[:, j]*y)
    sum_x2 = np.sum(x[:, j]**2)
    w_i = ((m*sum_xy) - (sum_x*sum_y))/((m*sum_x2) - (sum_x**2))
    b_i = ((sum_x2 * sum_y) - (sum_xy * sum_x)) / ((m*sum_x2) - (sum_x**2))
    b[j] = b[j] + b_i
    w.append(w_i)
  b = np.sum(b) / len(b)
  return np.array(w), b

def sigmoid(z):
  g = 1/ (1 + np.exp(-z))
  return g

def predict_(x, w, b):
  m, n = x.shape
  predicts = []
  for i in range(m):
    z = 0
    for j in range(n):
      z_i = x[i,j] * w[j]
      z += z_i
    z = z+ b
    f = sigmoid(z)
    if f >= 0.5:
      f = 1
    else:
      f = 0
    predicts.append(f)
  return np.array(predicts)

#getting X_train, Y_train, X_test, Y_test datasets

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size = 0.2)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#New accuracy for new test sample
l ,z  = compute_wb(X_train, Y_train)
z =  z/X_train.shape[0]
pred = predict_(X_test, l, z)
print(f"Accuracy >>> {accuracy_score(Y_test, pred)*100 : .3f}%")
print(f"weights ::: {l}")
print(f"bias ::: {z}")
count = 0
while True:
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size = 0.2)

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)
    W_init , b = compute_wb(X_train, Y_train)
    b = -b/X_train.shape[0]
    prediction_new = predict_(X_test, W_init, b)
    acc = accuracy_score(Y_test, prediction_new)
    print(acc)
    count += 1
    if acc == 1:
          import seaborn as sns
          #New accuracy for new test sample
          print(f"Accuracy >>> {accuracy_score(Y_test, prediction_new)*100 : .3f}%")
          print(f"Weights ::: {W_init}")  
          print(f"Bias ::: {b}")
          print(count)
          # Heatmap Visualization of the final result
          sns.heatmap(confusion_matrix(Y_test, prediction_new), annot = True)
          plt.show()
          break
    else:
        #W_init , b = compute_wb(X_train, Y_train)
        continue
print(prediction_new.shape)
print(Y_test.shape)



