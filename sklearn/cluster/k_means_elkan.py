import numpy as np

from ..metrics import euclidean_distances
from .k_means_ import _tolerance


def d(a, b):
    return np.linalg.norm(a - b)


def assign_labels(X, C, center_distances=None):
    # assigns closest center to X
    # uses triangle inequality
    if center_distances is None:
        center_distances = euclidean_distances(C) / 2.
    centers, distances = [], []
    for x in X:
        # assign first cluster center
        c_x = 0
        d_c = d(x, C[0])
        for j, c in enumerate(C):
            #print("d_c: %f" % d_c)
            #print("d(d_c, c'): %f" % center_distances[c_x, j])
            if d_c > center_distances[c_x, j]:
                dist = d(x, c)
                if dist < d_c:
                    d_c = dist
                    c_x = j
        centers.append(c_x)
        distances.append(d_c)
    return np.array(centers), np.array(distances)


def k_means_elkan(X, n_clusters, init, tol=1e-4, max_iter=30, verbose=False):
    #initialize
    tol = _tolerance(X, tol)
    centers = init
    n_samples = X.shape[0]
    n_centers = centers.shape[0]
    center_distances = euclidean_distances(centers) / 2.
    lower_bounds = np.zeros((n_samples, n_centers))
    labels, upper_bounds = assign_labels(
        X, centers, center_distances=center_distances)
    # make bounds tight for current labelss
    lower_bounds[np.arange(n_samples), labels] = upper_bounds
    bounds_tight = np.ones(n_samples, dtype=np.bool)
    for iteration in xrange(max_iter):
        distance_next_center = np.sort(center_distances, axis=0)[1]
        points_to_update = distance_next_center[labels] < upper_bounds
        for point_index in np.where(points_to_update)[0]:
            # check other update conditions
            for center_index, center in enumerate(centers):
                if (center_index != labels[point_index]
                        and (upper_bounds[point_index] >
                             lower_bounds[point_index, center_index])
                        and (upper_bounds[point_index] >
                             center_distances[center_index,
                                              labels[point_index]])):
                    # update distance to center
                    if not bounds_tight[point_index]:
                        upper_bounds[point_index] = \
                            d(X[point_index], centers[labels[point_index]])
                        lower_bounds[point_index, labels[point_index]] = \
                            upper_bounds[point_index]
                        bounds_tight[point_index] = True
                    # check for relabels
                    if (upper_bounds[point_index]
                            > lower_bounds[point_index, center_index]
                            or (upper_bounds[point_index] >
                                center_distances[labels[point_index],
                                                 center_index])):
                        distance = d(X[point_index], center)
                        lower_bounds[point_index, center_index] = distance
                        if distance < upper_bounds[point_index]:
                            labels[point_index] = center_index
                            upper_bounds[point_index] = distance

        # compute new centers
        new_centers = np.zeros_like(centers)
        for center_index in xrange(n_centers):
            new_centers[center_index] = np.mean(X[labels == center_index],
                                                axis=0)
        bounds_tight = np.zeros(n_samples, dtype=np.bool)

        # compute distance each center moved
        center_shift = np.sqrt(np.sum((centers - new_centers) ** 2, axis=1))
        # update bounds accordingly
        lower_bounds = np.maximum(lower_bounds - center_shift, 0)
        upper_bounds = upper_bounds + center_shift[labels]
        # reassign centers
        centers = new_centers
        if verbose:
            print('Iteration %i, inertia %s'
                  % (iteration, np.sum((X - centers[labels]) ** 2)))

        if np.sum(center_shift) < tol:
            print("center shift within tolerance")
            break
    return centers, labels
