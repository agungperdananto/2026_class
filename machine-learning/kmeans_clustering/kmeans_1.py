# -*- coding: utf-8 -*-
"""
K-Means (clustering) with scikit-learn
2 input (Income, SpendingScore), no label

@author: Agung Perdananto
"""
# import Library
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Import Data
customer_data = pd.read_csv('Customer_Data.csv')

# independent variables only (unsupervised learning has no dependent variable)
X = customer_data.iloc[:, :2].values

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow method to choose the number of clusters
inertias = []
for k in range(1, 9):
    model = KMeans(n_clusters=k, n_init=10, random_state=0)
    model.fit(X_scaled)
    inertias.append(model.inertia_)

plt.plot(range(1, 9), inertias, 'o-', c='g')
plt.title('Elbow Method')
plt.xlabel('number of clusters (K)')
plt.ylabel('inertia')
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters=5, n_init=10, random_state=0)
labels = kmeans.fit_predict(X_scaled)
centroids = scaler.inverse_transform(kmeans.cluster_centers_)

print('Inertia: ', kmeans.inertia_)
print('Centroids:')
print(centroids.round(1))

# Cluster profiling
customer_data['Cluster'] = labels
print(customer_data.groupby('Cluster').mean().round(1))

# Visualizing the clusters
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
for k in range(5):
    plt.scatter(X[labels == k, 0], X[labels == k, 1], c=colors[k], label=f'cluster {k}')
plt.scatter(centroids[:, 0], centroids[:, 1], c='k', marker='X', s=200, label='centroid')
plt.title('Customer Clusters')
plt.xlabel('income (million Rp / year)')
plt.ylabel('spending score (1-100)')
plt.legend()
plt.show()
