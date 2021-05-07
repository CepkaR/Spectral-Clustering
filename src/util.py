def acc(y_true, y_pred):
    import numpy as np
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    nmi = normalized_mutual_info_score
    ari = adjusted_rand_score

    """
    https://github.com/XifengGuo/DEC-keras/blob/master/metrics.py
    
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment as linear_assignment
    ind = np.asarray(linear_assignment(w.max() - w)).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def load_data(data, random_state=0):
    import tensorflow as tf
    import scipy
    import numpy as np
    from sklearn.utils import shuffle
    from sklearn import datasets

    if data == 'reuters':
        X = scipy.sparse.load_npz(
            'data/reuters_preprocessed/reutersidf_data.npz')
        y = np.load('data/reuters_preprocessed/reutersidf_target.npy')
        X, y = shuffle(X, y, random_state=random_state)
        X = X.astype('float32')

    elif data == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(
            path="mnist.npz")
        X = np.concatenate((x_train, x_test))
        y = np.concatenate((y_train, y_test))
        X, y = shuffle(X, y, random_state=random_state)
        X = X.reshape(-1, 28*28).astype('float32') / 255.

    elif data == 'fashion_mnist':
        (x_train, y_train), (x_test,
                             y_test) = tf.keras.datasets.fashion_mnist.load_data()
        X = np.concatenate((x_train, x_test))
        y = np.concatenate((y_train, y_test))
        X, y = shuffle(X, y, random_state=random_state)
        X = X.reshape(-1, 28*28).astype('float32') / 255.

    elif data == 'iris':
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target

        X, y = shuffle(X, y, random_state=random_state)
    else:
        raise ValueError('Dataset name is invalid.')

    return X, y
