# Author: Lars Buitinck <L.J.Buitinck@uva.nl>
# License: BSD 3 clause

from collections import Mapping
from operator import itemgetter

import numpy as np
import scipy.sparse as sp

from ..base import BaseEstimator, TransformerMixin
from ..externals import six
from ..externals.six.moves import xrange
from ..utils import atleast2d_or_csr, tosequence
from .base import vocabulary_transform, vocabulary_fit_transform


def _tosequence(X):
    """Turn X into a sequence or ndarray, avoiding a copy if possible."""
    if isinstance(X, Mapping):  # single sample
        return [X]
    else:
        return tosequence(X)


class DictVectorizer(BaseEstimator, TransformerMixin):
    """Transforms lists of feature-value mappings to vectors.

    This transformer turns lists of mappings (dict-like objects) of feature
    names to feature values into Numpy arrays or scipy.sparse matrices for use
    with scikit-learn estimators.

    When feature values are strings, this transformer will do a binary one-hot
    (aka one-of-K) coding: one boolean-valued feature is constructed for each
    of the possible string values that the feature can take on. For instance,
    a feature "f" that can take on the values "ham" and "spam" will become two
    features in the output, one signifying "f=ham", the other "f=spam".

    Features that do not occur in a sample (mapping) will have a zero value
    in the resulting array/matrix.

    Parameters
    ----------
    dtype : callable, optional
        The type of feature values. Passed to Numpy array/scipy.sparse matrix
        constructors as the dtype argument.
    separator: string, optional
        Separator string used when constructing new features for one-hot
        coding.
    sparse: boolean, optional.
        Whether transform should produce scipy.sparse matrices.
        True by default.

    Attributes
    ----------
    `feature_names_` : list
        A list of length n_features containing the feature names (e.g., "f=ham"
        and "f=spam").

    `vocabulary_` : dict
        A dictionary mapping feature names to feature indices.

    Examples
    --------
    >>> from sklearn.feature_extraction import DictVectorizer
    >>> v = DictVectorizer(sparse=False)
    >>> D = [{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}]
    >>> X = v.fit_transform(D)
    >>> X
    array([[ 2.,  0.,  1.],
           [ 0.,  1.,  3.]])
    >>> v.inverse_transform(X) == \
        [{'bar': 2.0, 'foo': 1.0}, {'baz': 1.0, 'foo': 3.0}]
    True
    >>> v.transform({'foo': 4, 'unseen_feature': 3})
    array([[ 0.,  0.,  4.]])

    See also
    --------
    DictVectorizer : performs vectorization in a similar as this class,
      but using a hash table instead of only a hash function.
    sklearn.preprocessing.OneHotEncoder : handles nominal/categorical features
      encoded as columns of integers.
    """

    def __init__(self, dtype=np.float64, separator="=", sparse=True):
        self.dtype = dtype
        self.separator = separator
        self.sparse = sparse

    def fit(self, X, y=None):
        """Learn a list of feature name -> indices mappings.

        Parameters
        ----------
        X : Mapping or iterable over Mappings
            Dict(s) or Mapping(s) from feature names (arbitrary Python
            objects) to feature values (strings or convertible to dtype).
        y : (ignored)

        Returns
        -------
        self
        """

        feature_names = set(f for x in X for f, v in self._encode_features(x))
        feature_names = sorted(feature_names)
        self.vocabulary_ = dict((f, i) for i, f in enumerate(feature_names))
        self.feature_names_ = feature_names
        return self

    def _encode_features(self, x):
        for f, v in six.iteritems(x):
            if isinstance(v, six.string_types):
                f = "%s%s%s" % (f, self.separator, v)
                v = 1
            yield f, v

    def fit_transform(self, X, y=None):
        """Learn a list of feature name -> indices mappings and transform X.

        Like fit(X) followed by transform(X).

        Parameters
        ----------
        X : Mapping or iterable over Mappings
            Dict(s) or Mapping(s) from feature names (arbitrary Python
            objects) to feature values (strings or convertible to dtype).
        y : (ignored)

        Returns
        -------
        Xa : {array, sparse matrix}
            Feature vectors; always 2-d.

        Notes
        -----
        If `sparse` is not set, this method requires two passes over X, so it
        materializes X in memory.
        """
        if not self.sparse:
            X = _tosequence(X)
            self.fit(X)
            return self.transform(X)

        vocabulary, Xt = vocabulary_fit_transform(X, self._encode_features,
                                                  dtype=self.dtype)
        self.vocabulary_ = dict(vocabulary)
        # XXX: repeats sort unnecessarily, but may be faster than iteritems:
        self.feature_names_ = sorted(vocabulary)
        return Xt

    def inverse_transform(self, X, dict_type=dict):
        """Transform array or sparse matrix X back to feature mappings.

        X must have been produced by this DictVectorizer's transform or
        fit_transform method; it may only have passed through transformers
        that preserve the number of features and their order.

        In the case of one-hot/one-of-K coding, the constructed feature
        names and values are returned rather than the original ones.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Sample matrix.
        dict_type : callable, optional
            Constructor for feature mappings. Must conform to the
            collections.Mapping API.

        Returns
        -------
        D : list of dict_type objects, length = n_samples
            Feature mappings for the samples in X.
        """
        X = atleast2d_or_csr(X)     # COO matrix is not subscriptable
        n_samples = X.shape[0]

        names = self.feature_names_
        dicts = [dict_type() for _ in xrange(n_samples)]

        if sp.issparse(X):
            for i, j in zip(*X.nonzero()):
                dicts[i][names[j]] = X[i, j]
        else:
            for i, d in enumerate(dicts):
                for j, v in enumerate(X[i, :]):
                    if v != 0:
                        d[names[j]] = X[i, j]

        return dicts

    def transform(self, X, y=None):
        """Transform feature->value dicts to array or sparse matrix.

        Named features not encountered during fit or fit_transform will be
        silently ignored.

        Parameters
        ----------
        X : Mapping or iterable over Mappings, length = n_samples
            Dict(s) or Mapping(s) from feature names (arbitrary Python
            objects) to feature values (strings or convertible to dtype).
        y : (ignored)

        Returns
        -------
        Xa : {array, sparse matrix}
            Feature vectors; always 2-d.
        """
        X = _tosequence(X)
        return vocabulary_transform(X, self._encode_features, self.vocabulary_,
                                    dtype=self.dtype, sparse=self.sparse)

    def get_feature_names(self):
        """Returns a list of feature names, ordered by their indices.

        If one-of-K coding is applied to categorical features, this will
        include the constructed feature names but not the original ones.
        """
        return self.feature_names_

    def restrict(self, support, indices=False):
        """Restrict the features to those in support.

        Parameters
        ----------
        support : array-like
            Boolean mask or list of indices (as returned by the get_support
            member of feature selectors).
        indices : boolean, optional
            Whether support is a list of indices.
        """
        if not indices:
            support = np.where(support)[0]

        names = self.feature_names_
        new_vocab = {}
        for i in support:
            new_vocab[names[i]] = len(new_vocab)

        self.vocabulary_ = new_vocab
        self.feature_names_ = [f for f, i in sorted(six.iteritems(new_vocab),
                                                    key=itemgetter(1))]

        return self
