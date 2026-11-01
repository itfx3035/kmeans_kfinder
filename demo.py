'''KMeans k finder demo.

Uses the sample dataset from sample_dataset.py (ground truth k = 4) and scikit-learn's
Iris dataset (ground truth k = 3, overlapping classes, so the elbow methods may
disagree with the true number of clusters).
'''

from sklearn.datasets import load_iris

from kmeans_kfinder import KMeansKFinder
from sample_dataset import get_dataset


def run(name, X, ground_truth_k):
    kmf = KMeansKFinder(X, max_k=30, random_state=13)
    best_k = kmf.find_best_k()

    print(f'\nDataset: {name} (ground truth k={ground_truth_k})')
    print(kmf.summary())


def main():
    X, truth = get_dataset('clean_blobs')
    run('clean_blobs', X, truth)

    iris = load_iris()
    run('iris', iris.data, 3)


if __name__ == '__main__':
    main()
