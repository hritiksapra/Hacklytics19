import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import copy

dataset = pd.read_csv("gun.csv")

dataset.latitude = dataset.latitude.astype(float).fillna(0.0)
dataset.longitude = dataset.longitude.astype(float).fillna(0.0)

f1 = list(dataset["latitude"])
f2 = list(dataset["longitude"])
X = np.array(list(zip(f1, f2)))

kmeans = KMeans(n_clusters=100)
kmeans = kmeans.fit(X)
labels = kmeans.predict(X)
centroids = kmeans.cluster_centers_

plt.scatter(X[:,0],X[:,1], c=kmeans.labels_, cmap='rainbow')
plt.xlim(24,50)
plt.ylim(-130,-62)
plt.xlabel('Latitude')
plt.ylabel("Longitude")
plt.show()