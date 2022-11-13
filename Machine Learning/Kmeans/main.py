
#%%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, y = make_blobs(n_samples=100, centers=6, 
cluster_std=0.60, random_state=0)

plt.plot(X[:,0], X[:,1])

WCSS = []

for k in range(1,15):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    WCSS.append(kmeans.inertia_)
    
# %%
plt.plot(range(1,15),WCSS)
plt.xlabel("Number of K Value(Cluster)")
plt.ylabel("WCSS")
plt.grid()
plt.show()
# %%
kmeans =KMeans(n_clusters = 6 ,max_iter=300, random_state= 1)
clusters = kmeans.fit_predict(X) 
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
# %%
h = 0.1
x_min, x_max = X[:,0].min(), X[:,0].max()
y_min, y_max = X[:,1].min(), X[:,1].max()
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()]) 
Z = Z.reshape(xx.shape)
 
plt.figure(1 , figsize = (15 , 7) )
plt.clf()
plt.imshow(Z , interpolation='nearest', 
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap = plt.cm.Pastel1, origin='lower')
 
plt.scatter(x=X[:,0], y=X[:,1], c=labels, s=100)
 
plt.scatter(x=centroids[:,0], y=centroids[:,1],s=300 , c='red')
 
plt.ylabel('x') , plt.xlabel('y')
plt.grid()
plt.title("Clustering")
plt.show()
# %%
