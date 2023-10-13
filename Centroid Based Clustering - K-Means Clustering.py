#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#run before importing Kmeans
import os
os.environ["OMP_NUM_THREADS"]='1'

#importing the dataset
dataset=pd.read_csv("Mall_Customers.csv")
dataset

#series of scatterplots for each pair of variables and a histogram for each variable
sns.pairplot(dataset.iloc[:,[2,3,4]])

#importing StandardScaler from scikit-learn library
from sklearn.preprocessing import StandardScaler

# select all rows (:) and only the columns at index 3 and 4
X=dataset.iloc[:,[3,4]].values

# creating an instance of the StandardScaler class and storing it in the variable sc_X
sc_X=StandardScaler()

#standardizing the values stored in X (mean =0, sd =1)
X=sc_X.fit_transform(X)

#using elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
#intializing an empty list
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('No of clusters')
plt.ylabel('WCSS')
plt.show()


#fitting kmeans to the dataset
kmeans=KMeans(n_clusters=5,init='k-means++',random_state=42)

#fits the K-Means model to your data X and assigns each data point to one of the clusters. 
#The fit_predict method does both the fitting and the prediction. 
#After running this line, y_kmeans will be a NumPy array containing cluster assignments for each data point in X
y_kmeans=kmeans.fit_predict(X)
y_kmeans


#Visualizing the clusters
#sets the figure size for your plot, making it an 8x8-inch square.
plt.figure(figsize=(8,8))

#X[y_kmeans==0, 0]: This selects the data points from your dataset X that belong to Cluster 1
#(cluster with label 0) based on the y_kmeans array. 
#X[y_kmeans==0, 0] represents the x-coordinates of the data points in Cluster 1.
#X[y_kmeans==0, 1]: This selects the y-coordinates of the data points in Cluster 1.
#It corresponds to the second column of your dataset.
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c='red',label='Cluster1')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c='blue',label='Cluster2')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c='green',label='Cluster3')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100,c='cyan',label='Cluster4')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=100,c='magenta',label='Cluster5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')
plt.title('cluster of customers')
plt.xlabel('annual income(scaled)')
plt.ylabel('spending income(scaled)')
plt.legend()
plt.show()


# In[ ]:




