# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Author: Andreas Mueller
#
# Licence: BSD 3 clause

import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport sqrt

from ..metrics import euclidean_distances
from .k_means_ import _tolerance
from ._k_means import _centers_dense


cdef double d(double[:] a, double[:] b) nogil:
    cdef double result, tmp
    result = 0
    cdef int i
    for i in range(a.shape[0]):
        tmp = (a[i] - b[i])
        result += tmp * tmp
    return sqrt(result)


cdef assign_labels(double[:, :] X, double[:, :] centers, double[:, :]
                   center_distances, int[:] labels, double[:] distances):
    print("start assign labels")
    # assigns closest center to X
    # uses triangle inequality
    cdef int n_samples = X.shape[0]
    cdef int n_centers = centers.shape[0]
    cdef double[:] x, c
    cdef double d_c, dist
    cdef int c_x, j, sample
    for sample in range(n_samples):
        # assign first cluster center
        c_x = 0
        x = X[sample]
        d_c = d(x, centers[0])
        for j in range(n_centers):
            if d_c > center_distances[c_x, j]:
                c = centers[j]
                dist = d(x, c)
                if dist < d_c:
                    d_c = dist
                    c_x = j
        labels[sample] = c_x
        distances[sample] = d_c
    print("end assign labels")


def k_means_elkan(X_, n_clusters, init, float tol=1e-4, int max_iter=30, verbose=False):
    #initialize
    tol = _tolerance(X_, tol)
    centers_ = init
    cdef double[:, :] centers = centers_
    cdef double[:, :] X = X_
    cdef int n_samples = X.shape[0]
    cdef int n_centers = centers.shape[0]
    cdef int point_index, center_index, label
    cdef float upper_bound, distance
    cdef double[:, :] center_distances = euclidean_distances(centers_) / 2.
    cdef double[:, :] lower_bounds = np.zeros((n_samples, n_centers))
    cdef double[:] x, distance_next_center
    labels_ = np.empty(n_samples, dtype=np.int32)
    cdef int[:] labels = labels_
    upper_bounds_ = np.empty(n_samples, dtype=np.float)
    cdef double[:] upper_bounds = upper_bounds_
    assign_labels(X, centers, center_distances, labels, upper_bounds)
    # make bounds tight for current labelss
    for point_index in range(n_samples):
        lower_bounds[point_index, labels[point_index]] = upper_bounds[point_index]
    cdef np.uint8_t[:] bounds_tight = np.ones(n_samples, dtype=np.uint8)
    cdef np.uint8_t[:] points_to_update = np.zeros(n_samples, dtype=np.uint8)
    for iteration in range(max_iter):
        print("start iteration")
        # we could find the closest center in O(n) but
        # this does not seem to be the bottleneck
        distance_next_center = np.sort(center_distances, axis=0)[1]
        print("done sorting")
        for point_index in range(n_samples):
            upper_bound = upper_bounds[point_index]
            label = labels[point_index]
            if distance_next_center[label] >= upper_bound:
                continue
            x = X[point_index]
            # check other update conditions
            for center_index in range(n_centers):
                if (center_index != label
                        and (upper_bound > lower_bounds[point_index, center_index])
                        and (upper_bound > center_distances[center_index, label])):
                    # update distance to center
                    if not bounds_tight[point_index]:
                        upper_bound = d(x, centers[label])
                        lower_bounds[point_index, label] = upper_bound
                        bounds_tight[point_index] = 1
                    # check for relabels
                    if (upper_bound > lower_bounds[point_index, center_index]
                            or (upper_bound > center_distances[label, center_index])):
                        distance = d(x, centers[center_index])
                        lower_bounds[point_index, center_index] = distance
                        if distance < upper_bound:
                            label = center_index
                            upper_bound = distance
            labels[point_index] = label
            upper_bounds[point_index] = upper_bound
        print("end inner loop")
        # compute new centers
        new_centers = _centers_dense(X_, labels_, n_centers, upper_bounds_)
        bounds_tight = np.zeros(n_samples, dtype=np.uint8)

        # compute distance each center moved
        center_shift = np.sqrt(np.sum((centers_ - new_centers) ** 2, axis=1))
        # update bounds accordingly
        lower_bounds = np.maximum(lower_bounds - center_shift, 0)
        upper_bounds = upper_bounds + center_shift[labels_]
        # reassign centers
        centers = new_centers
        centers_ = new_centers
        # update between-center distances
        center_distances = euclidean_distances(centers_) / 2.
        if verbose:
            print('Iteration %i, inertia %s'
                  % (iteration, np.sum((X - centers_[labels]) ** 2)))

        if np.sum(center_shift) < tol:
            print("center shift within tolerance")
            break
    return centers_, labels
