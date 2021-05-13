from spectral_methods import SpectralClustering
from sklearn.cluster import KMeans
import numpy as np
import time
from util import acc
from sklearn.metrics import normalized_mutual_info_score
from util import load_data
import pandas as pd
from itertools import product


def run_experiment(data, num_of_experiments, params):

    def param_permutation(params):
        return [dict(zip(params.keys(), values)) for values in product(*params.values())]

    overal_results = {}
    params = param_permutation(params)
    columns = pd.MultiIndex.from_product([data.keys(), ["NMI", "ACC"]])
    index = ["-".join([str(v) for v in param.values()]) for param in params]
    table = pd.DataFrame(index=index, columns=columns)
    for param in params:
        for dataset, n_clusters in data.items():
            X, y = load_data(dataset)
            NMI = []
            ACC = []
            start_time = time.time()
            for _ in range(num_of_experiments):
                sc = SpectralClustering(n_clusters=n_clusters, n_neighbors=int(
                    np.log(X.shape[0])*10)).set_params(**param)

                y_labels = sc.fit(X).labels_

                NMI.append(
                    np.round(normalized_mutual_info_score(y, y_labels), 3))
                ACC.append(np.round(acc(y, y_labels), 3))

            print(
                f"runtime of {num_of_experiments} experiments, {dataset} dataset: {np.round(time.time() - start_time,3)}s")
            print(f"{param=}")

            table.loc["-".join([str(v) for v in param.values()]), (dataset, "NMI")
                      ] = f'{np.around(np.mean(np.array(NMI)),decimals=3)} +- {np.around(np.std(np.array(NMI)),decimals=3)}'
            table.loc["-".join([str(v) for v in param.values()]), (dataset, "ACC")
                      ] = f'{np.around(np.mean(np.array(ACC)),decimals=3)} +- {np.around(np.std(np.array(ACC)),decimals=3)}'

    return table


if __name__ == "__main__":
    # dataset:n_clusters
    data = {'iris': 3, 'mnist': 10, 'fashion_mnist': 10, 'reuters': 4}

    # parameter n_neighbors was set in run_experiment(), n_neighbors = int(10*ln(n_samples))
    params = {"sigma": ["max", "mean"], "type_of_laplacian": [
        "rw", "sym", "clasic"], "affinity": ["k-hNNG"]}

    num_of_experiments = 1

    results_table = run_experiment(data, num_of_experiments, params)
    results_table.to_csv("experiments_results")
