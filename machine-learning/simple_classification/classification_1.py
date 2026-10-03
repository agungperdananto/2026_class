# -*- coding: utf-8 -*-
"""
Simple Logistic Regression (classification) with scikit-learn
2 input (HoursStudied, PracticeTests) and 1 output (Passed)

@author: Agung Perdananto
"""
# import Library
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Import Data
study_data = pd.read_csv('Study_Data.csv')

# separate dependent variable and independent variable
X = study_data.iloc[:, :-1].values

y = study_data.iloc[:, -1].values

#splitting the dataset into the training set dan Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 0)

# Fitting Simple Logistic Regression to training set
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Evaluation
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

# Decision boundary: b + m1*x1 + m2*x2 = 0  ->  x2 = -(b + m1*x1) / m2
b = classifier.intercept_[0]
m1, m2 = classifier.coef_[0]
x1_line = np.linspace(0, 10.5, 100)
x2_line = -(b + m1 * x1_line) / m2

# Visualizing training results
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c='r', label='failed')
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c='b', label='passed')
plt.plot(x1_line, x2_line, c='g')
plt.ylim(-0.5, 10.5)
plt.title('Passed vs Hours Studied & Practice Tests (Training data)')
plt.xlabel('hours studied')
plt.ylabel('practice tests')
plt.legend()
plt.show()

# Visualizing test results
plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], c='r', label='failed')
plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], c='b', label='passed')
plt.plot(x1_line, x2_line, c='g')
plt.ylim(-0.5, 10.5)
plt.title('Passed vs Hours Studied & Practice Tests (Test data)')
plt.xlabel('hours studied')
plt.ylabel('practice tests')
plt.legend()
plt.show()
