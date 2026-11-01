'''Shared helpers for the experiment scripts in this folder.

The experiments measure which documented behaviour is worth keeping; they are not
production code and are not imported by the library. Run them from the repository root:

    python3 experiments/method1_variants.py
    python3 experiments/convention.py
    python3 experiments/max_k_sensitivity.py
'''

import os
import sys
from collections import Counter
from math import sqrt

import numpy as np
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kmeans_kfinder import KMeansKFinder        # noqa: E402
from sample_dataset import DATASETS, get_dataset  # noqa: E402

SEEDS = list(range(1, 6))
MAX_K = 30
CLEAR_SEPARATION = 2.0


def suite():
    '''Datasets with a known ground truth k: (name, X, truth, separation_or_None).

    Synthetic blobs with varying cluster separation plus the project sample datasets.
    A dataset is "clear" (separation >= CLEAR_SEPARATION) when ground truth k is
    meaningful for elbow heuristics; on ambiguous data any answer can be defended.
    '''
    rng = np.random.default_rng(2024)
    datasets = []
    for separation in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 10.0):
        for k in (2, 3, 4, 6):
            centers = rng.normal(scale=separation * 3, size=(k, 2))
            X = np.vstack([centers[i] + rng.normal(scale=1.0, size=(100, 2)) for i in range(k)])
            datasets.append((f'sep{separation}_k{k}', X, k, separation))
    for name in DATASETS:
        X, truth = get_dataset(name)
        if X.shape[0] >= 30:
            datasets.append((f'sample:{name}', X, truth, None))
    return datasets


def is_clear(separation):
    return separation is not None and separation >= CLEAR_SEPARATION


def vote(values):
    '''Majority vote, largest k wins ties (the documented rule).'''
    values = [v for v in values if v is not None]
    if not values:
        return None
    counts = Counter(values)
    top = max(counts.values())
    return max(k for k, count in counts.items() if count == top)


def finder_for(X, seed, max_k=MAX_K):
    '''KMeansKFinder instance with the stats table built for one seed.'''
    finder = KMeansKFinder(X, max_k=max_k, random_state=seed)
    finder.find_best_k()
    return finder


def distances(X, seed, max_k=MAX_K):
    '''(k, distance) pairs as the stats table is built from them.'''
    out = []
    for k in range(1, max_k + 1):
        distance = sqrt(KMeans(n_clusters=k, random_state=seed).fit(X).inertia_)
        out.append((k, distance))
        if distance == 0:
            break
    return out
