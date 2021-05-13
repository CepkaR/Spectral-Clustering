# Spectral clustering

Abstract:

Spectral clustering is an important method of cluster analysis based on the spectral theory of matrices. In this thesis we deal with the relations of spectral clustering to several mathematical areas, for example to graph theory, perturbation theory and Markov chains. We prove the Rayleigh-Ritz theorem, which is important not only for the spectral clustering method, but can also be used in the method of principal component analysis and other methods that use eigenvalues and eigenvectors. Furthermore, in this thesis we deal with spectral clustering from the point of view of applications, where we present experimental results on non-trivial real data, and we also discuss the appropriate setting of hyperparameters.

## Requirements
This code was written in python 3.8.8 with the following packages:

* numpy 1.19.2
* scipy 1.6.2
* scikit-learn 0.24.1
* pynndescent 0.5.2

* tensorflow 2.3.0 (only for download MNIST and Fashion-MNIST dataset)
* pandas 1.2.3 (only for store experimental results)

If you want to run experiments.py script with REUTERS dataset, you schould download it by cd data; ./get_data.sh and prepocess it by make_reuters_dataset.py.

## Example usage
...
