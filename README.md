# Low-Dimensional Embedding Techniques Comparison
This project aims to reimplement and compare two modifications of the tSNE algorithm for low-dimensional embedding of high-dimensional data. We will create several synthetic datasets, apply the embedding techniques, and compare their results based on different metrics.

## Getting Started
You can install the dependencies using the following command:

```shell
pip install -r requirements.txt
```
## Datasets
We will generate several synthetic datasets using the scikit-learn library. The datasets will have different properties, such as varying levels of noise, different shapes, and varying degrees of clustering.

## Metrics
To compare the performance of the embedding techniques, we will use the following metrics:

* Silhouette score: measures how well the samples are clustered and separated from each other.
* Homogeneity score: measures how well the samples within each cluster belong to the same class.
* Mutual information: measures the mutual dependence between the clustering and the true labels.

We will also visualize the embeddings using scatter plots and heatmaps to gain a better understanding of their properties.

## Baseline Embedding Techniques
We will compare the performance of the tSNE modifications to the following baseline techniques:

* PCA (Principal Component Analysis)
* MDS (Multidimensional Scaling)
* UMAP (Uniform Manifold Approximation and Projection)
* tSNE

## Implemented Techniques
We will reimplement and apply the following modifications of the tSNE algorithm:

* dtSNE (Dynamic t-Distributed Stochastic Neighbor Embedding)
* JEDI (Joint Embedding and Distribution Inference)

## Results
We will present the results of the comparison in a table and discuss the findings in the Conclusion section of the report.

## Conclusion
Based on the metrics and visualization, we will conclude which embedding techniques have improved the lower-dimensional representations of the data and which have failed to do so. We will also discuss the strengths and weaknesses of each method and suggest potential areas for future research.
