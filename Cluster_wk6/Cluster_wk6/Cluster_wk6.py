import graphlab
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import time
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

'''Check GraphLab Create version'''
from distutils.version import StrictVersion
assert (StrictVersion(graphlab.version) >= StrictVersion('1.8.5')), 'GraphLab Create must be version 1.8.5 or later.'

wiki = graphlab.SFrame('D:\\ML_Learning\\UW_Cluster\\week6\\people_wiki.gl\\')
wiki['tf_idf'] = graphlab.text_analytics.tf_idf(wiki['text'])

import sys
sys.path.insert(0,'D:\\ML_Learning\\UW_Cluster\\week6\\')
from em_utilities import sframe_to_scipy # converter

# This will take about a minute or two.
tf_idf, map_index_to_word = sframe_to_scipy(wiki, 'tf_idf')

from sklearn.preprocessing import normalize
tf_idf = normalize(tf_idf)

def bipartition(cluster, maxiter=400, num_runs=4, seed=None):
    '''cluster: should be a dictionary containing the following keys
                * dataframe: original dataframe
                * matrix:    same data, in matrix format
                * centroid:  centroid for this particular cluster'''
    
    data_matrix = cluster['matrix']
    dataframe   = cluster['dataframe']
    
    # Run k-means on the data matrix with k=2. We use scikit-learn here to simplify workflow.
    kmeans_model = KMeans(n_clusters=2, max_iter=maxiter, n_init=num_runs, random_state=seed, n_jobs=1)
    kmeans_model.fit(data_matrix)
    centroids, cluster_assignment = kmeans_model.cluster_centers_, kmeans_model.labels_
    
    # Divide the data matrix into two parts using the cluster assignments.
    data_matrix_left_child, data_matrix_right_child = data_matrix[cluster_assignment==0], \
                                                      data_matrix[cluster_assignment==1]
    
    # Divide the dataframe into two parts, again using the cluster assignments.
    cluster_assignment_sa = graphlab.SArray(cluster_assignment) # minor format conversion
    dataframe_left_child, dataframe_right_child     = dataframe[cluster_assignment_sa==0], \
                                                      dataframe[cluster_assignment_sa==1]
        
    
    # Package relevant variables for the child clusters
    cluster_left_child  = {'matrix': data_matrix_left_child,
                           'dataframe': dataframe_left_child,
                           'centroid': centroids[0]}
    cluster_right_child = {'matrix': data_matrix_right_child,
                           'dataframe': dataframe_right_child,
                           'centroid': centroids[1]}
    
    return (cluster_left_child, cluster_right_child)

wiki_data = {'matrix': tf_idf, 'dataframe': wiki} # no 'centroid' for the root cluster
left_child, right_child = bipartition(wiki_data, maxiter=100, num_runs=6, seed=1)

def display_single_tf_idf_cluster(cluster, map_index_to_word):
    '''map_index_to_word: SFrame specifying the mapping betweeen words and column indices'''
    
    wiki_subset   = cluster['dataframe']
    tf_idf_subset = cluster['matrix']
    centroid      = cluster['centroid']
    
    # Print top 5 words with largest TF-IDF weights in the cluster
    idx = centroid.argsort()[::-1]
    for i in xrange(5):
        print('{0:s}:{1:.3f}'.format(map_index_to_word['category'][idx[i]], centroid[idx[i]])),
    print('')
    
    # Compute distances from the centroid to all data points in the cluster.
    distances = pairwise_distances(tf_idf_subset, [centroid], metric='euclidean').flatten()
    # compute nearest neighbors of the centroid within the cluster.
    nearest_neighbors = distances.argsort()
    # For 8 nearest neighbors, print the title as well as first 180 characters of text.
    # Wrap the text at 80-character mark.
    for i in xrange(8):
        text = ' '.join(wiki_subset[nearest_neighbors[i]]['text'].split(None, 25)[0:25])
        print('* {0:50s} {1:.5f}\n  {2:s}\n  {3:s}'.format(wiki_subset[nearest_neighbors[i]]['name'],
              distances[nearest_neighbors[i]], text[:90], text[90:180] if len(text) > 90 else ''))
    print('')
display_single_tf_idf_cluster(left_child, map_index_to_word)
display_single_tf_idf_cluster(right_child, map_index_to_word)

athletes = left_child
non_athletes = right_child
# Bipartition the cluster of athletes
left_child_athletes, right_child_athletes = bipartition(athletes, maxiter=100, num_runs=6, seed=1)
display_single_tf_idf_cluster(left_child_athletes, map_index_to_word)
display_single_tf_idf_cluster(right_child_athletes, map_index_to_word)
baseball            = left_child_athletes
ice_hockey_football = right_child_athletes
left_child_ihs, right_child_ihs = bipartition(ice_hockey_football, maxiter=100, num_runs=6, seed=1)
display_single_tf_idf_cluster(left_child_ihs, map_index_to_word)
display_single_tf_idf_cluster(right_child_ihs, map_index_to_word)

# Bipartition the cluster of non-athletes
left_child_non_athletes, right_child_non_athletes = bipartition(non_athletes, maxiter=100, num_runs=6, seed=1)
display_single_tf_idf_cluster(left_child_non_athletes, map_index_to_word)
display_single_tf_idf_cluster(right_child_non_athletes, map_index_to_word)
male_non_athletes = left_child_non_athletes
female_non_athletes = right_child_non_athletes

# Bipartition the cluster of non-athletes
left_child_male_non_athletes, right_child_male_non_athletes = bipartition(male_non_athletes, maxiter=100, num_runs=6, seed=1)
display_single_tf_idf_cluster(left_child_male_non_athletes, map_index_to_word)
display_single_tf_idf_cluster(right_child_male_non_athletes, map_index_to_word)

# Bipartition the cluster of non-athletes
left_child_female_non_athletes, right_child_female_non_athletes = bipartition(female_non_athletes, maxiter=100, num_runs=6, seed=1)
display_single_tf_idf_cluster(left_child_female_non_athletes, map_index_to_word)
display_single_tf_idf_cluster(right_child_female_non_athletes, map_index_to_word)