# -*- coding: utf-8 -*-
"""
Decision Tree (classification) with scikit-learn
2 input (Income, CreditScore) and 1 output (Approved)

@author: Agung Perdananto
"""
# import Library
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Import Data
loan_data = pd.read_csv('Loan_Data.csv')

# separate dependent variable and independent variable
X = loan_data.iloc[:, :-1].values

y = loan_data.iloc[:, -1].values

#splitting the dataset into the training set dan Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 0)

# Fitting Decision Tree to training set
classifier = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Evaluation
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

# The learned tree as text
print(export_text(classifier, feature_names=['Income', 'CreditScore']))

# Visualizing the tree
plt.figure(figsize=(14, 6))
plot_tree(classifier, feature_names=['Income', 'CreditScore'], class_names=['rejected', 'approved'],
          filled=True, rounded=True)
plt.show()

# Decision regions: predict every point of a grid
x1, x2 = np.meshgrid(np.linspace(0, 22, 300), np.linspace(380, 870, 300))
grid_pred = classifier.predict(np.c_[x1.ravel(), x2.ravel()]).reshape(x1.shape)

# Visualizing training results
plt.contourf(x1, x2, grid_pred, levels=[-0.5, 0.5, 1.5], colors=['r', 'b'], alpha=0.2)
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c='r', label='rejected')
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c='b', label='approved')
plt.title('Approved vs Income & Credit Score (Training data)')
plt.xlabel('income (million Rp / month)')
plt.ylabel('credit score')
plt.legend()
plt.show()

# Visualizing test results
plt.contourf(x1, x2, grid_pred, levels=[-0.5, 0.5, 1.5], colors=['r', 'b'], alpha=0.2)
plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], c='r', label='rejected')
plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], c='b', label='approved')
plt.title('Approved vs Income & Credit Score (Test data)')
plt.xlabel('income (million Rp / month)')
plt.ylabel('credit score')
plt.legend()
plt.show()
