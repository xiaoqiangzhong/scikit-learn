from collections import defaultdict
from array import array

import numpy as np
import scipy.sparse as sp

from ..utils import tosequence
from ..utils import atleast2d_or_csc
from ..externals import six


def vocabulary_transform(X, get_features, vocabulary, dtype=np.int64,
                         sparse=True):
    if sparse:
        # Sanity check: Python's array has no way of explicitly requesting
        # the signed 32-bit integers that scipy.sparse needs, so we use the
        # next best thing: typecode "i" (int). However, if that gives
        # larger or smaller integers than 32-bit ones, np.frombuffer screws
        # up.
        assert array("i").itemsize == 4, (
            "sizeof(int) != 4 on your platform; please report this at"
            " https://github.com/scikit-learn/scikit-learn/issues and"
            " include the output from platform.platform() in your bug"
            " report")

        indices = array("i")
        indptr = array("i", [0])
        # XXX we could change values to an array.array as well, but it
        # would require (heuristic) conversion of dtype to typecode...
        values = []

        for x in X:
            feats = list(get_features(x))
            print(feats)
            for f, v in feats:
                try:
                    f = vocabulary[f]
                except KeyError:
                    continue
                indices.append(f)
                values.append(dtype(v))
            indptr.append(len(indices))

        if len(indptr) == 0:
            raise ValueError("Sample sequence X is empty.")

        if len(indices) > 0:
            # workaround for bug in older NumPy:
            # http://projects.scipy.org/numpy/ticket/1943
            indices = np.frombuffer(indices, dtype=np.intc)
        indptr = np.frombuffer(indptr, dtype=np.intc)
        if not vocabulary:
            raise ValueError('empty vocabulary')
        shape = (len(indptr) - 1, len(vocabulary))
        Xa = sp.csr_matrix((values, indices, indptr),
                           shape=shape, dtype=dtype)
        Xa.sum_duplicates()

    else:
        X = tosequence(X)
        Xa = np.zeros((len(X), len(vocabulary)), dtype=dtype)

        for i, x in enumerate(X):
            for f, v in get_features(x):
                try:
                    f = vocabulary[f]
                except KeyError:
                    continue
                Xa[i, f] = dtype(v)

    return Xa


def vocabulary_fit_transform(X, get_features, dtype=np.int64, sort=True):
    # Dict that adds a new value when a new vocabulary item is seen
    vocabulary = defaultdict(int)
    vocabulary.default_factory = vocabulary.__len__
    Xt = vocabulary_transform(X, get_features, vocabulary, dtype)
    vocabulary = dict(vocabulary)

    if sort:
        Xt = atleast2d_or_csc(Xt)
        sorted_features = sorted(six.iteritems(vocabulary))
        map_index = np.empty(len(sorted_features), dtype=np.int32)
        for new_val, (term, old_val) in enumerate(sorted_features):
            map_index[new_val] = old_val
            vocabulary[term] = new_val
        Xt = Xt[:, map_index]
    return vocabulary, Xt
