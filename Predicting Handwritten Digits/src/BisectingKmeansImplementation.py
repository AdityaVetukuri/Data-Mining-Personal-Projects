/*
Name : Aditya Varma Vetukuri
GID  : G01213246 
*/
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import Normalizer
from tqdm import tqdm

center=[]
sse=[]
i=1
clusters=[]
centroids=[]
final_labels = []
def Kmeans_clustering(data, n_clusters=2, rseed=2):
	global clusters
	global centroids
	rng = np.random.RandomState(rseed)
	index = rng.permutation(data.shape[0])[:n_clusters]
	centroids = data[index]
	while True:
		temp=[]
		dist=cdist(data, centroids)
		for i in dist:
		    temp.append(np.where(i==i.min())[0][0])
		labels=np.asarray(temp)
		new_centroids = np.array([data[labels == index].mean(0) for index in range(n_clusters)])
		if np.all(centroids == new_centroids):
		    break
		centroids = new_centroids
	for i in range(n_clusters):
		clusters.append(data[labels==i])




bisected_clusters=[]
count = 0
def bisect_KMeans(data):
	global i
	if(i<=20):
		i+=1
		Kmeans_clustering(data)
		a=clusters[0]

		b=clusters[1]
		clusters.clear()
		sse.append(sum(np.min(cdist(data, centroids), axis=1)) / data.shape[0])

		if len(a)>len(b):
			bisected_clusters.append(b)
			center.append(centroids[1])
			bisect_KMeans(np.array(a).astype(np.float))
		else:
			bisected_clusters.append(a)
			center.append(centroids[0])
			bisect_KMeans(np.array(b).astype(np.float))
	else:
		return

X=[]


def getData(filename):
    for line in open(filename):
        temp = line.split(',')
        list = []
        for t in temp:
            list.append(float(t))
        X.append(list)
    return X
X = getData("part2image.txt")
X=np.array(X).astype(np.float)

X = Normalizer(norm='max').fit_transform(X)

tsne_model = TSNE()

tsne_data  = tsne_model.fit_transform(X)


bisect_KMeans(tsne_data)
y_pred = np.zeros(X.shape[0],dtype = int)
k = 0
for cluster in bisected_clusters:
	for x in tqdm(cluster):

		i = np.argwhere([all(_) for _ in X == x])[0][0]
		y_pred[i] = k
	k += 1


k=range(2,22)
plt.plot(k, sse, 'bx-')
plt.xlabel('k')
plt.ylabel('SSE')
plt.title('Bisecting K means showing the optimal k')
plt.show()