import numpy as np
from sklearn.metrics import silhouette_score, homogeneity_score, mutual_info_score


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
