import myplots as myplt
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import scipy.io as io
from scipy.cluster.hierarchy import dendrogram, linkage  #

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u


# ----------------------------------------------------------------------
"""
Part 1: 
Evaluation of k-Means over Diverse Datasets: 
In the first task, you will explore how k-Means perform on datasets with diverse structure.
"""

# Fill this function with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_kmeans(dataset, n_clusters):
    data, _ = dataset 
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    kmeans = KMeans(n_clusters=n_clusters, init='random', random_state=42)
    kmeans.fit(data_scaled)
    
    return kmeans.labels_


def compute():
    answers = {}

    """
    A.	Load the following 5 datasets with 100 samples each: noisy_circles (nc), noisy_moons (nm), blobs with varied variances (bvv), Anisotropicly distributed data (add), blobs (b). Use the parameters from (https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html), with any random state. (with random_state = 42). Not setting the correct random_state will prevent me from checking your results.
    """

    # Dictionary of 5 datasets. e.g., dct["nc"] = [data, labels]
    # 'nc', 'nm', 'bvv', 'add', 'b'. keys: 'nc', 'nm', 'bvv', 'add', 'b' (abbreviated datasets)
    
    noisy_circles = datasets.make_circles(n_samples=100, factor=.5, noise=.05, random_state=42)
    noisy_moons = datasets.make_moons(n_samples=100, noise=.05, random_state=42)
    blobs = datasets.make_blobs(n_samples=100, random_state=42)
    varied = datasets.make_blobs(n_samples=100, cluster_std=[1.0, 2.5, 0.5], random_state=42)
    
    # Generate Anisotropicly distributed data
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(blobs[0], transformation)
    aniso = (X_aniso, blobs[1])

    datasets = {
        "nc": noisy_circles,
        "nm": noisy_moons,
        "bvv": varied,
        "add": aniso,
        "b": blobs
        }
    
    for key in datasets.keys():
            datasets[key] = (StandardScaler().fit_transform(datasets[key][0]), datasets[key][1])
    
    # Organizing datasets in a dictionary as specified

    dct = answers["1A: datasets"] = datasets

    """
   B. Write a function called fit_kmeans that takes dataset (before any processing on it), i.e., pair of (data, label) Numpy arrays, and the number of clusters as arguments, and returns the predicted labels from k-means clustering. Use the init='random' argument and make sure to standardize the data (see StandardScaler transform), prior to fitting the KMeans estimator. This is the function you will use in the following questions. 
    """

    # dct value:  the `fit_kmeans` function
    dct = answers["1B: fit_kmeans"] = fit_kmeans


    """
    C.	Make a big figure (4 rows x 5 columns) of scatter plots (where points are colored by predicted label) with each column corresponding to the datasets generated in part 1.A, and each row being k=[2,3,5,10] different number of clusters. For which datasets does k-means seem to produce correct clusters for (assuming the right number of k is specified) and for which datasets does k-means fail for all values of k? 
    
    Create a pdf of the plots and return in your report. 
    """
    ks = [2, 3, 5, 10]

    plt.figure(figsize= (25, 20))

    dataset_names = {"nc": "Noisy Circles",
                     "nm": "Noisy Moons",
                     "bvv": "Varied Variances",
                     "add": "Anisotropic",
                     "b": "Blobs"}
    
    for i,k in enumerate(ks):
        for j, (dataset_abbr, dataset) in enumerate(datasets.items()):
            predicted_labels = fit_kmeans(dataset, k)
            plt.subplot(len(ks), len(datasets), i*len(datasets) + j + 1)
            plt.scatter(dataset[0][:, 0], dataset[0][:, 1], c=predicted_labels, s=50, cmap='viridis')
            plt.title(f"{dataset_names[dataset_abbr]} (k={k})")

    plt.show()

    # dct value: return a dictionary of one or more abbreviated dataset names (zero or more elements) 
    # and associated k-values with correct clusters.  key abbreviations: 'nc', 'nm', 'bvv', 'add', 'b'. 
    # The values are the list of k for which there is success. Only return datasets where the list of cluster size k is non-empty.
    answers = ["1C: cluster failures"] = ["nc",
                                          "nm"]

    
    dct = answers["1C: cluster successes"] = {"bvv": [3],
                                              "add": [3],
                                              "b": [3]} 

    # dct value: return a list of 0 or more dataset abbreviations (list has zero or more elements, 
    # which are abbreviated dataset names as strings)
    
    """
    D. Repeat 1.C a few times and comment on which (if any) datasets seem to be sensitive to the choice of initialization for the k=2,3 cases. You do not need to add the additional plots to your report.

    Create a pdf of the plots and return in your report. 
    """

    # dct value: list of dataset abbreviations
    # Look at your plots, and return your answers.
    # The plot is part of your report, a pdf file name "report.pdf", in your repository.

    dct = answers["1D: datasets sensitive to initialization"] = ["Noisy Circles and Noisy Moons are affected by how they are initially set up due to their unique shapes and the way they overlap. These datasets are made up of points that create complex and interlocking designs, challenging the spherical cluster assumption foundational to techniques such as k-means. This situation means that where the centroids are first positioned can significantly impact the clustering outcome, as the points do not have distinct divisions."]
    
    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part1.pkl", "wb") as f:
        pickle.dump(answers, f)
