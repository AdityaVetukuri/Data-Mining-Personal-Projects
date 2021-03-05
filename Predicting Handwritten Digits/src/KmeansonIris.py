/*
Name : Aditya Varma Vetukuri
GID  : G01213246 
*/
import sys
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances_argmin
import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer


def distance_1(p1, p2):
    return np.sum((p1 - p2) ** 2)


# initialisation algorithm
def initialize(data, k):

    centroids = []
    centroids.append(data[np.random.randint(
        data.shape[0]), :])
    for c_id in range(k - 1):


        dist = []
        for i in range(data.shape[0]):
            point = data[i, :]
            d = sys.maxsize


            for j in range(len(centroids)):
                temp_dist = distance_1(point, centroids[j])
                d = min(d, temp_dist)
            dist.append(d)

        dist = np.array(dist)
        next_centroid = data[np.argmax(dist), :]
        centroids.append(next_centroid)
        dist = []

    return centroids

def KMeans_clustering(X, n_clusters, rseed=2):

    rng = np.random.RandomState(rseed)
    X = np.array(X)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = initialize(X,n_clusters)
    centers = np.array(centers)

    while True:

        temp = []
        dist = distance.cdist(X, centers, 'cosine')
        for i in dist:
            temp.append(np.where(i == i.min())[0][0])
        labels = np.asarray(temp)



        new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])


        if np.all(centers == new_centers):
            break
        centers = new_centers
        # t=0
    return labels


X = []

for line in open("iris_test.txt"):
    l=line.split("\n")
    for a in l:
        if a!="":
            temp=a.split(" ")
            temp[0]=float(temp[0])
            temp[1]=float(temp[1])
            temp[2]=float(temp[2])
            temp[3]=float(temp[3])
            X.append(temp)

X = np.array(X).astype(np.float)

X = Normalizer(norm='max').fit_transform(X)

labels = KMeans_clustering(X,3)

f = open("lastpart1.txt", "w")
for a in labels:
    f.write(str(a + 1))
    f.write("\n")

f.close()
