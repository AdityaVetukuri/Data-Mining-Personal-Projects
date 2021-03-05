/*
Name : Aditya Varma Vetukuri
GID  : G01213246 
*/
import sys
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from skimage.transform import downscale_local_mean
from sklearn.manifold import TSNE
from sklearn.preprocessing import Binarizer, Normalizer


def distance_centroid(p1, p2):
    return np.sum((p1 - p2) ** 2)


# initialisation algorithm
def initialize(data, k):

    centroids = []
    centroids.append(data[np.random.randint(
        data.shape[0]), :])

    ## compute remaining centroids
    for centroid_idx in range(k - 1):

        ## initialize a list to store distances of data
        ## points from nearest centroid
        distances = []
        for i in range(data.shape[0]):
            point = data[i, :]
            d = sys.maxsize

            ## compute distance of 'point' from each of the previously
            ## selected centroid and store the minimum distance
            for j in range(len(centroids)):
                temp_distance = distance_centroid(point, centroids[j])
                d = min(d, temp_distance)
            distances.append(d)


        distances = np.array(distances)
        next_centroid = data[np.argmax(distances), :]
        centroids.append(next_centroid)
        distances = []

    return centroids

def KMeans_clustering(X, n_clusters, rseed=2):
    rng = np.random.RandomState(rseed)
    X = np.array(X)
    centers = initialize(X, n_clusters)
    centers = np.array(centers)
    while True:
        temp = []
        distances = distance.cdist(X, centers, 'euclidean')
        for dis in distances:
            temp.append(np.where(dis == dis.min())[0][0])
        labels = np.asarray(temp)
        new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])
        if np.all(centers == new_centers):
            break
        centers = new_centers
    return centers

X = []

def getData(filename):
    for line in open(filename):
        temp = line.split(',')
        list = []
        for t in temp:
            list.append(float(t))
        X.append(list)
    return X
X = getData("part2image.txt")

X = np.array(X).astype(np.float)

bin = Binarizer()

X = bin.fit_transform(X)

#Reshaping the image to 28 x 28 pixels for downscaling

x_reshaped = np.reshape(X , (10740,28,28))

#downscaling the image by a factor of 2
image_downscaled = downscale_local_mean(x_reshaped, (1,2, 2))

#Re shaping the image by flattening
image_reshaped = np.reshape(image_downscaled,(10740,196))

X = Normalizer().fit_transform(image_reshaped)
#

tsne_model = TSNE()

tsne_data  = tsne_model.fit_transform(X)

sse=[]
k= range(2,21,2)
for i in range(2,21,2):
    centers=KMeans_clustering(tsne_data,i)
    sse.append(sum(np.min(cdist(tsne_data, centers,'cosine'), axis=1)) / X.shape[0])

plt.plot(k, sse, 'bx-')
plt.xlabel('k')
plt.ylabel('SSE')
plt.title('OPTIMAL K WITH TSNE with COSINE')
plt.show()

