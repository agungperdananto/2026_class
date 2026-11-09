# -*- coding: utf-8 -*-
"""
K-Means (clustering) with PCA using scikit-learn
7 input (StudyHours, Attendance, AssignmentScore, ExamScore, Extracurricular, SocialMedia, SleepHours), no label

@author: Agung Perdananto
"""
# import Library
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Import Data
student_data = pd.read_csv('Student_Data.csv')
feature_names = list(student_data.columns)

# independent variables only (unsupervised learning has no dependent variable)
X = student_data.values

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA with all components: how much information every component keeps
pca_full = PCA().fit(X_scaled)
print('Explained variance ratio: ', pca_full.explained_variance_ratio_.round(3))

plt.bar(range(1, 8), pca_full.explained_variance_ratio_, color='lightsteelblue')
plt.plot(range(1, 8), np.cumsum(pca_full.explained_variance_ratio_), 'o-', c='g')
plt.title('Scree Plot')
plt.xlabel('principal component')
plt.ylabel('explained variance ratio')
plt.show()

# Reduce 7 features to 2 principal components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print('Information kept: ', pca.explained_variance_ratio_.sum().round(3))
print(pd.DataFrame(pca.components_.T, index=feature_names, columns=['PC1', 'PC2']).round(3))

# Elbow method to choose the number of clusters
inertias = []
for k in range(1, 9):
    model = KMeans(n_clusters=k, n_init=10, random_state=0)
    model.fit(X_pca)
    inertias.append(model.inertia_)

plt.plot(range(1, 9), inertias, 'o-', c='g')
plt.title('Elbow Method')
plt.xlabel('number of clusters (K)')
plt.ylabel('inertia')
plt.show()

# Fitting K-Means to the PCA result
kmeans = KMeans(n_clusters=4, n_init=10, random_state=0)
labels = kmeans.fit_predict(X_pca)

# Cluster profiling with the original features
student_data['Cluster'] = labels
print(student_data.groupby('Cluster').mean().round(1))

# Visualizing the clusters in the PCA space
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
for k in range(4):
    plt.scatter(X_pca[labels == k, 0], X_pca[labels == k, 1], c=colors[k], label=f'cluster {k}')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='k', marker='X', s=200, label='centroid')
plt.title('Student Clusters (PCA)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()
