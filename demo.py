''' 
    KMeans k finder demo
'''

from sklearn.datasets import load_iris
from kmeans_kfinder import KMeansKFinder

test_data = load_iris()['data'][:,0:5]

kmf_object = KMeansKFinder(test_data, max_k=30, random_state=13)
kmf_object.find_best_k()

print('Full output:', kmf_object.best_k_opts)
print('Best K:', kmf_object.best_k)

