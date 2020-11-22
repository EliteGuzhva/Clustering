import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans, OPTICS, AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_rand_score, homogeneity_score
from sklearn.metrics.cluster import completeness_score, v_measure_score

from util import *
from data_loader import DataLoader

DATASET_IDX = 1       # 1..6
METHOD = 'k-means'    # ['k-means', 'optics', 'agglo']
METRIC = 'euclidean'  # ['euclidean', 'manhattan']

RANDOM_STATE = 42
MIN_SAMPLES = 30

if DATASET_IDX == 1:
    DATASET = '1'
    N_CLUSTERS = 3
elif DATASET_IDX == 2:
    DATASET = '2'
    N_CLUSTERS = 3
elif DATASET_IDX == 3:
    DATASET = '3'
    N_CLUSTERS = 3
    MIN_SAMPLES = 5
elif DATASET_IDX == 4:
    DATASET = '4'
    N_CLUSTERS = 2
    MIN_SAMPLES = 5
elif DATASET_IDX == 5:
    DATASET = 'breast_cancer'
    N_CLUSTERS = 2
elif DATASET_IDX == 6:
    DATASET = 'chinese_mnist'
    N_CLUSTERS = 2
else:
    print("Wrong DATASET_IDX")
    exit()

# load data
sep()
print("DATASET:", DATASET)

dl = DataLoader(DATASET, verbose=1)
X, y = dl.load()

# clustering
sep()
print("METHOD:", METHOD)
print("METRIC:", METRIC)

if METHOD == 'k-means':
    if METRIC == 'manhattan':
        method = KMedoids(N_CLUSTERS, metric=METRIC, random_state=RANDOM_STATE)
    elif METRIC == 'euclidean':
        method = KMeans(N_CLUSTERS, random_state=RANDOM_STATE)
elif METHOD == 'optics':
    method = OPTICS(min_samples=MIN_SAMPLES, metric=METRIC, n_jobs=-1)
elif METHOD == 'agglo':
    if METRIC == 'euclidean':
        method = AgglomerativeClustering(N_CLUSTERS, linkage='ward')
    elif METRIC == 'manhattan':
        method = AgglomerativeClustering(N_CLUSTERS, affinity=METRIC,
                                         linkage='single')
method = method.fit(X)
clusters = method.labels_

# metrics
sep()
print("QUALITY METRICS")

rand = adjusted_rand_score(y, clusters)
print(f"Rand: {rand:.3f}")
homo = homogeneity_score(y, clusters)
print(f"Homogeneity score: {homo:.3f}")
comp = completeness_score(y, clusters)
print(f"Completeness score: {comp:.3f}")
v_meas = v_measure_score(y ,clusters)
print(f"V measure score: {v_meas:.3f}")

# visualize data
sep()
print("Visualizing...")

if X.shape[1] > 2:
    embedding = TSNE(n_components=2, random_state=RANDOM_STATE)
    X = embedding.fit_transform(X)

plt.tricontourf(X[:, 0], X[:, 1], clusters,
                cmap='viridis', alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', s=100)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow')

print("Done!")

plt.show()

