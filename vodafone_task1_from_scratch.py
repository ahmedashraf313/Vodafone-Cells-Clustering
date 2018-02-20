#import the libraries
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances


#get the dataset
dataset= pd.read_csv('./Vodafone/ML Project/task1.csv')


#get the columns related to clustring
dataset=dataset.iloc[:,:].values


####################apply kmeans algorithm########################################
#1-intialize centroids
def get_initial_centroids(data, k, seed=None):
        '''Randomly choose k data points as initial centroids'''
        if seed is not None: # useful for obtaining consistent results
            np.random.seed(seed)
            
        # number of data points    
        #n = data.shape[0] 
        n=len(data)    
         
        # Pick K indices from range [0, N).
        rand_indices = np.random.randint(0, n, k)
        
        # Keep centroids as dense format, as many entries will be nonzero due to averaging.
        # As long as at least one document in a cluster contains a word,
        # it will carry a nonzero weight in the TF-IDF vector of the centroid.
        centroids = data[rand_indices,:]
        
        return centroids
    
    #centroids=get_initial_centroids(dataset[:,1:],k,0)
    
#2-Assign clusters
def assign_clusters(data, centroids):
        
        # Compute distances between each data point and the set of centroids:
        # Fill in the blank (RHS only)
        distances_from_centroids = pairwise_distances(data, centroids, metric='euclidean')
        
        # Compute cluster assignments for each data point:
        # Fill in the blank (RHS only)
        cluster_assignment = np.argmin(distances_from_centroids, axis=1)
        
        return cluster_assignment
    
    #dataset[:,0]=assign_clusters(dataset[:,1:],centroids)
    
    
#3-compute and place the new centroids
def revise_centroids(data, k):
        new_centroids = []
        for i in range(k):
            # Select all data points that belong to cluster i. Fill in the blank (RHS only)
            member_data_points = data[data[:,0] == i]
            # Compute the mean of the data points. Fill in the blank (RHS only)
            centroid = member_data_points.mean(axis=0)
            
            # Convert numpy.matrix type to numpy.ndarray type
            #centroid = centroid.A1
            new_centroids.append(centroid)
        new_centroids = np.array(new_centroids)
        
        return new_centroids[:,1:]
    
    #new_centroids=revise_centroids(dataset,k)
    
    
    
    
#4-compute heterogenity
def compute_heterogeneity(data, k, centroids):
        
        heterogeneity = 0.0
        for i in range(k):
            
            # Select all data points that belong to cluster i. Fill in the blank (RHS only)
            member_data_points = data[data[:,0] == i]
            member_data_points=member_data_points[:,1:]
            
            # Compute distances from centroid to data points (RHS only)
            distances = pairwise_distances(member_data_points, [centroids[i]], metric='euclidean')
            squared_distances = distances**2
            heterogeneity += np.sum(squared_distances)
            
        return heterogeneity


def kmeans(data,k,seed,step,centroids=None):
    if (step==0):
        centroids=get_initial_centroids(data[:,1:],k,0)
        dataset[:,0]=assign_clusters(dataset[:,1:],centroids)
        new_centroids=revise_centroids(data,k)
        #4-check if the algorithm convergence or not
        if((new_centroids==centroids).all()):
            heterogeneity=compute_heterogeneity(data,k,new_centroids)
            wcss.append(heterogeneity)

        else:
           kmeans(data,k,0,1,new_centroids)
    
    elif(step==1):
        dataset[:,0]=assign_clusters(dataset[:,1:],centroids)
        new_centroids=revise_centroids(data,k)
        #4-check if the algorithm convergence or not
        if((new_centroids==centroids).all()):
            heterogeneity=compute_heterogeneity(data,k,new_centroids)
            wcss.append(heterogeneity)

        else:
           kmeans(data,k,0,1,new_centroids)
        


 
#choose the best number of clusters
wcss=[]
for k in range(2,11):
      
       kmeans(dataset,k,0,0)
    
    
#draw the relation between the WCSS and K number of clusters
import matplotlib.pyplot as plt
plt.plot(range(2,11),wcss)
plt.show()    


#from the plot that the best k is 5
kmeans(dataset,5,0,0)

