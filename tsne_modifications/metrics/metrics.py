import numpy as np
from sklearn.metrics import silhouette_score, homogeneity_score, mutual_info_score
from sklearn.neighbors import NearestNeighbors

def local_reconstruction_correlation(X, n_neighbors=100):
    """
    Calculates the correlation between high- and low-dimensional distances of each point
    with its 100 closest neighbors using local reconstruction.
    
    Parameters:
    X (array-like, shape (n_samples, n_features)): Input data.
    n_neighbors (int): Number of neighbors to use for the reconstruction.
    
    Returns:
    correlation (float): The correlation between high- and low-dimensional distances.
    """
    # Fit a nearest neighbors model to the input data
    knn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
    
    # Find the k nearest neighbors for each point
    distances, indices = knn.kneighbors(X)
    
    # Remove the first column (it's just the point itself)
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    
    # Compute the pairwise distances between the points
    pairwise_distances = np.sqrt(((X[:, np.newaxis] - X)**2).sum(axis=2))
    
    # Compute the pairwise distances between each point and its reconstructed neighbors
    reconstructed_distances = np.sqrt(((X[:, np.newaxis] - X[indices])**2).sum(axis=2))
    
    # Compute the correlation between the two sets of distances
    correlation = np.corrcoef(pairwise_distances.flatten(), reconstructed_distances.flatten())[0, 1]
    
    return correlation

def evaluate_embedding(X, y_true, embedding):
    """
    Evaluates the performance of an embedding technique using three metrics:
    silhouette score, homogeneity score, and mutual information.
    
    Parameters:
    X (array-like, shape (n_samples, n_features)): The high-dimensional data.
    y_true (array-like, shape (n_samples,)): The true labels of the samples.
    embedding (array-like, shape (n_samples, n_components)): The low-dimensional embedding.
    
    Returns:
    A dictionary containing the three metric scores.
    """
    n_clusters = len(np.unique(y_true))
    sil_score = silhouette_score(X, y_true)
    hom_score = homogeneity_score(y_true, np.argmax(embedding, axis=1))
    mi_score = mutual_info_score(y_true, np.argmax(embedding, axis=1))
    
    return {'silhouette_score': sil_score,
            'homogeneity_score': hom_score,
            'mutual_info_score': mi_score}
