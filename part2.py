from pprint import pprint

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

import myplots as myplt
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import scipy.io as io
from scipy.cluster.hierarchy import dendrogram, linkage  #
from sklearn.cluster import KMeans
# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

# ----------------------------------------------------------------------
"""
Part 2
Comparison of Clustering Evaluation Metrics: 
In this task you will explore different methods to find a good value for k
"""

# Fill this function with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_kmeans(dataset, n_clust):
    data, _ = dataset 
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    kmeans = KMeans(n_clusters=n_clust, init='random', random_state=42)
    kmeans.fit(data_scaled)

    sse = 0
    clusters_center = kmeans.cluster_centers_
    labels = kmeans.labels_
    for i in range(len(data_scaled)):
        sse += np.square(np.linalg.norm(data_scaled[i] - clusters_center[labels[i]]))

    return sse



def compute():
    # ---------------------
    answers = {}

    """
    A.	Call the make_blobs function with following parameters :(center_box=(-20,20), n_samples=20, centers=5, random_state=12).
    """

    data, labels = make_blobs(center_box = (-20, 20), n_samples = 20, centers = 5, random_state = 12)

    # dct: return value from the make_blobs function in sklearn, expressed as a list of three numpy arrays
    dct = answers["2A: blob"] = [data, labels, np.zeros(20)]

    """
    B. Modify the fit_kmeans function to return the SSE (see Equations 8.1 and 8.2 in the book).
    """

    # dct value: the `fit_kmeans` function
    dct = answers["2B: fit_kmeans"] = fit_kmeans

    """
    C.	Plot the SSE as a function of k for k=1,2,….,8, and choose the optimal k based on the elbow method.
    """
    sse_vals = []
    for k in range(1,9):
        sse = fit_kmeans([data, labels], k)
        sse_vals.append([k, sse])

    plt.figure(figsize =(10,10))
    plt.plot([k for k, sse in sse_vals], [sse for k, sse in sse_vals], marker = "+")
    plt.xlabel("Num of Clusters (k)")
    plt.ylabel("SSE")
    plt.title("SSE vs K")
    plt.grid(True)
    plt.show()

    # dct value: a list of tuples, e.g., [[0, 100.], [1, 200.]]
    # Each tuple is a (k, SSE) pair
    dct = answers["2C: SSE plot"] = [[k, sse] for k, sse in sse_vals]

    """
    D.	Repeat part 2.C for inertia (note this is an attribute in the kmeans estimator called _inertia). Do the optimal k’s agree?
    """
    inert_vals = []
    for k in range(1,9):
        scaler = StandardScaler()
        data_scaled =  scaler.fit_transform(data)
        kmeans = KMeans(n_clusters = k, init = 'random', random_state = 42)
        kmeans.fit(data_scaled)
        inertia = kmeans.inertia_
        inert_vals.append([k , inertia])
    
    plt.figure(figsize = (10,10))
    plt.plot([k for k, inertia in inert_vals], [inertia for k, inertia in inert_vals])
    plt.xlabel("Num of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Inertia vs K")
    plt.grid(True)
    plt.show()

    # dct value has the same structure as in 2C
    dct = answers["2D: inertia plot"] = [[k, inertia] for k, inertia in inert_vals]

    # dct value should be a string, e.g., "yes" or "no"
    dct = answers["2D: do ks agree?"] = "yes"

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part2.pkl", "wb") as f:
        pickle.dump(answers, f)
