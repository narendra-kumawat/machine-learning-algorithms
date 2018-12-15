#Logistic Regression

#data preprocessing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset 
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2,3]] # metrics of features
Y = dataset.iloc[:, 4] # dependent variable

# Splitting dataset into the traning set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

# Feature Scalling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting Logistic Regression to the Training Set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap

x_set, y_set = X_train, Y_train

x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() -1, stop = x_set[:, 0].max()+1, step = 0.01), 
                     np.arange(start = x_set[:, 0].min() -1, stop = x_set[:, 0].max()+1, step = 0.01))

plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
        alpha = 0.75, cmap = ListedColormap(('red','green')))

plt.xlim(x1.min(), x1.max())
plt.xlim(x2.min(), x2.max())

# plotting points on the square 
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red','green'))(i), label =j)
    
plt.title("Logistic Regression (Test Set)")
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()