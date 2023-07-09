#kmeans clustering
#unlabeled dataset
#we use sse and no of cluster graph which create elbow type of graph
#elbow decide how many cluster we have to make
#sse fromula is sum(xi-ci)^2 c is centroid
#no of centroid is equal no of clusters
#new point distance check from all given centroids 
#the minumum distance will decide in which cluster the point will be moved
from sklearn.datasets import load_iris
import pandas as pd
data=load_iris()
x=pd.DataFrame(data.data)
y=pd.DataFrame(data.target)
print(y)

from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
model=KMeans(3)#no of clusters
model.fit(x,y)
ypred=model.predict(x)
#unique values
print(x.head())
center=model.cluster_centers_#predefined function to give all the centroid points
plt.scatter(x[0],x[3])
plt.scatter(center[:,0],center[:,3],c='r')

from sklearn.datasets import make_blobs
x,y=make_blobs(n_samples=1000,n_features=3,centers=2)

from sklearn.metrics import accuracy_score
kmodel=KMeans(2)
kmodel.fit(x)
ypred=kmodel.predict(x)
center=kmodel.cluster_centers_
plt.scatter(x[:,0],x[:,1])
plt.scatter(center[:,0],center[:,1])
plt.show()

plt.scatter(x[:,0],x[:,2])
plt.scatter(center[:,0],center[:,2])
plt.show()

#draw kmeans scratch
#1 random centroids create
#2 create clusters
#3 create new centroids
#4 pedict the cluster
import numpy as np
import matplotlib.pyplot as plt
class kmeans:
    def __init__(self,x,num_cluster):
        self.k=num_cluster
        self.max_iter=100
        self.num_samples,self.num_features=x.shape
    def centroidinitialize(self,x):
        centroids=np.zeros((self.k,self.num_features))#[[4.6,5.3,0.5,1.3],[0,0,0],[0,0,0]]
        for k in range(self.k):#3 times
            centroid=x[np.random.choice(range(self.num_samples))]#546[12,34,54]
            centroids[k]=centroid
        return centroids
    def clustercreation(self,x,centroids):
        clusters=[[] for _ in range(self.k)]#[[0,5,8,3,65,79,32],[67,87,90,54,5,14,2,56,54],[56,23,45,65,78]]
        for pid,p in enumerate(x):
            closest_centroid=np.argmin(np.sqrt(np.sum((p-centroids)**2,axis=1)))#0,1,2
            clusters[closest_centroid].append(pid)
        return clusters
    def newcreation(self,cluster,x):
        centroids=np.zeros((self.k,self.num_features))
        for idx,cluster in enumerate(cluster):
                new_centroid=np.mean(x[cluster])
                centroids[idx]=new_centroid
        return centroids
    #[[3,6,4,87,8]]                                   
    def predict_cluster(self,clusters,x):
            y_pred=np.zeros(self.num_samples)#[0,0,0,   1000values]
            for cidx,cluster in enumerate(clusters):#id will be 0,1,2 [[nvalue],[nvalue],[nvalue]]
                #nvalues are the index of x array
                for j in cluster:#j is the index of array x
                    y_pred[j]=cidx
            return y_pred             
    def fit(self,x):
        centroids=self.centroidinitialize(x)#3[23,45,67]
        for i in range(self.max_iter):
            clusters=self.clustercreation(x,centroids)
            prev_centroid=centroids#[10,20,30]
            centroids=self.newcreation(clusters,x)#3[10,20,30]
            differ=centroids-prev_centroid
            if(differ.any()):
                break
        y_pred=self.predict_cluster(clusters,x)
        print(centroids)
        plt.scatter(x[:,0],x[:,1],c='g')
        plt.scatter(centroids[:,0],centroids[:,1],c='r')
from sklearn.datasets import make_blobs
x,y=make_blobs(n_samples=1000,n_features=3,centers=2)
km=kmeans(x,3)
km.fit(x)

            
