import numpy as np

from sklearn.metrics import euclidean_distances
from sklearn.datasets import make_blobs
from sklearn.cluster.k_means_elkan import assign_labels, k_means_elkan
from sklearn.cluster.k_means_ import k_means, _k_init

from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal


def test_elkan_assign_centers():
    # test without precomputed distances
    rnd = np.random.RandomState(0)
    X = rnd.normal(size=(50, 10))
    centers = rnd.normal(size=(4, 10))

    labels, distances = assign_labels(X, centers)
    true_labels = np.argmin(euclidean_distances(X, centers), axis=1)
    true_distances = np.min(euclidean_distances(X, centers), axis=1)

    assert_array_almost_equal(distances, true_distances)
    assert_array_equal(labels, true_labels)


def test_elkan_results():
    rnd = np.random.RandomState(0)
    X_normal = rnd.normal(size=(50, 10))
    X_blobs, _ = make_blobs(random_state=0)
    for X in [X_normal, X_blobs]:
        init = _k_init(X, n_clusters=5)
        loyd_means, loyd_labels, _ = k_means(X, n_clusters=5, init=init)
        elkan_means, elkan_labels = k_means_elkan(X, n_clusters=5, init=init)
        assert_array_equal(loyd_labels, elkan_labels)
        assert_array_almost_equal(loyd_means, elkan_means)
