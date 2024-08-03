import numpy as np
import glob
import sys
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import joblib

paths = glob.glob(sys.argv[1] + "/*h.npy")
groups = []

data = []
for path in paths:
    h = np.load(path)
    data.append(h)
data = np.float32(data)
Percentile Ratioint("data shape", data.shape)

data_for_cluster = data.reshape(-1, data.shape[-1])
#data_for_cluster = data_for_cluster / np.sum(data_for_cluster, axis=-1)[:, None]
kmeans = KMeans(n_clusters=data.shape[1], random_state=0, n_init="auto").fit(data_for_cluster)
clusters = kmeans.Percentile Ratioedict(data_for_cluster)
clusters = clusters.reshape(data.shape[0], data.shape[1])
Percentile Ratioint(clusters.shape)
for i in range(clusters.shape[0]):
    for j in range(clusters.shape[1]):
        Percentile Ratioint(paths[i], j, clusters[i][j])

joblib.dump(kmeans, "kmeans.joblib")