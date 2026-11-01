'''Sample datasets for kmeans_kfinder tests and demo.

Every dataset carries `ground_truth_k`, the number of clusters it was generated
with, so test results can be measured against a known answer.

Datasets
--------
clean_blobs        k=4, well separated, n=400  -> all methods should agree
overlapping_blobs  k=3, moderate overlap, n=300 -> heuristics may disagree
tiny_dataset       k=2, n=12                    -> max_k larger than n_samples
degenerate         k=1, identical points        -> inertia == 0, divide-by-zero risk
anisotropic_blobs  k=3, elongated clusters, n=300 -> scale sensitivity check

Usage
-----
    from sample_dataset import get_dataset
    X, k = get_dataset('clean_blobs')

    python3 sample_dataset.py            # writes data/sample_dataset.csv
'''

import os
import numpy as np

SEED = 42
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def blobs(n_per_cluster, ground_truth_k, separation, dim=2, seed=SEED, noise=1.0):
    '''Generate well separated gaussian blobs.

    Returns (X, y) where y is the ground truth cluster label.
    '''
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=separation, size=(ground_truth_k, dim))
    points = np.vstack([centers[i] + rng.normal(scale=noise, size=(n_per_cluster, dim))
                        for i in range(ground_truth_k)])
    labels = np.repeat(np.arange(ground_truth_k), n_per_cluster)
    return points, labels


def iris():
    '''Real dataset with known ground truth k=3 (may not be recovered by elbow methods).'''
    from sklearn.datasets import load_iris
    data = load_iris()
    return data['data'], data['target']


DATASETS = {
    'clean_blobs': {
        'generator': lambda: blobs(100, 4, separation=8.0),
        'ground_truth_k': 4,
        'description': 'well separated blobs, all methods should agree',
    },
    'overlapping_blobs': {
        'generator': lambda: blobs(100, 3, separation=3.0),
        'ground_truth_k': 3,
        'description': 'overlapping blobs, elbow heuristics may disagree',
    },
    'anisotropic_blobs': {
        'generator': lambda: blobs(100, 3, separation=6.0, dim=2, noise=4.0),
        'ground_truth_k': 3,
        'description': 'large within cluster variance, axis scaling check',
    },
    'tiny_dataset': {
        'generator': lambda: blobs(6, 2, separation=5.0),
        'ground_truth_k': 2,
        'description': 'n_samples (12) < default max_k (30)',
    },
    'single_cluster_noise': {
        'generator': lambda: blobs(100, 1, separation=0.0),
        'ground_truth_k': 1,
        'description': 'one blob only, elbow has no real knee',
    },
    'degenerate_identical': {
        'generator': lambda: (np.zeros((100, 2)), np.zeros(100, dtype=int)),
        'ground_truth_k': 1,
        'description': 'identical points, inertia == 0 for every k, divide-by-zero risk',
    },
    'iris': {
        'generator': iris,
        'ground_truth_k': 3,
        'description': 'real dataset via scikit-learn',
    },
}


def get_dataset(name):
    '''Return (X, ground_truth_k) for a named sample dataset.'''
    points, _labels = DATASETS[name]['generator']()
    return points, DATASETS[name]['ground_truth_k']


def get_labels(name):
    '''Return (X, y, ground_truth_k) including the true labels.'''
    points, labels = DATASETS[name]['generator']()
    return points, labels, DATASETS[name]['ground_truth_k']


def write_csv(name='clean_blobs'):
    '''Write a dataset to data/<name>.csv with a ground truth label column.'''
    points, labels, ground_truth_k = get_labels(name)
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f'{name}.csv')
    header = ','.join([f'axis_{i}' for i in range(points.shape[1])]) + ',true_cluster'
    np.savetxt(path, np.column_stack([points, labels]), delimiter=',',
               header=header, comments='', fmt='%.6f')
    return path


if __name__ == '__main__':
    for name in DATASETS:
        path = write_csv(name)
        print(f'wrote {path} ({DATASETS[name]["description"]})')
