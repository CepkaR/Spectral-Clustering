# Spectral clustering

![](https://github.com/CepkaR/Spectral-Clustering/blob/main/spectral_clustering.png)
*Aeroplane background image is taken from this [link](https://www.newscientist.com/article/2255751-zero-emissions-hydrogen-plane-test-was-part-powered-by-fossil-fuels/).*

This is the accompanying code to our bachelor thesis. 
If you want to reproduce results from our bachelor thesis, run experiments.py script. This package can also be freely used 
as spectral clustering implementation for cluster analysis purpose (see Usage for examples).

### Abstract:
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

## Usage
```python
from spectral_methods import SpectralClustering
import numpy as np

# data
x = np.array([[0, 0], [0, 1], [1, 1],
              [5, 5], [5, 6], [6, 6]])

# define hyperparameters
params = {'n_clusters': 2, 'n_neighbors': 3,
          'sigma': 'max', 'affinity': 'k-hNNG',
          'type_of_laplacian': 'rw', 'knn_aprox': False,
          'eigen_aprox': False}

sc = SpectralClustering(**params).fit(x)

print(sc.labels_) # output: [1 1 1 0 0 0]
```

```python
from spectral_methods import SpectralClustering
import numpy as np
from util import load_data
from util import acc
from sklearn.metrics import normalized_mutual_info_score as nmi

# load MNIST dataset
x, y_true = load_data('mnist')

# define hyperparameters
params = {'n_clusters': 10, 'n_neighbors': int(
    np.log(X.shape[0])*10),
    'sigma': 'max', 'affinity': 'k-hNNG',
    'type_of_laplacian': 'rw', 'knn_aprox': True,
    'eigen_aprox': True}

# time: aprox 24 seconds
sc = SpectralClustering(**params).fit(x)

# evaluate clustering results (only if true labels is known)
print(nmi(y_true, sc.labels_))  # output: 0.6808
print(acc(y_true, sc.labels_))  # output: 0.6304
```
