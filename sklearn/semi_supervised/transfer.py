"""Transfer, inductive, or self-taught learning utility
"""
# Author: Joel Nothman
# Licence: BSD 3 clause

import functools

import numpy as np

from ..utils import safe_mask
from ..base import BaseEstimator


class Transfer(BaseEstimator):
    """Supervised learning using an unsupervised data transformation

    This implements a simple semi-supervised model, belonging to the family of
    transfer, indictive, or self-taught learning techniques. A transformation
    is learnt from unlabeled samples before being applied to train a
    supervised estimator.

    Parameters
    ----------
    transformer : estimator object implementing transformer interface
        An unsupervised transformer.

    estimator : estimator object
        A supervised estimator.

    unlabeled : label or callable (default -1)
        The value which should be assumed to be unlabeled, or a callable
        which creates a boolean mask of unlabeled values given an array of
        targets, such as `np.isnan`.

    iid : boolean (default False)
        If the unlabeled and labeled samples are independent and identically
        distributed (`iid=True`), then all samples will be used to train the
        transformer. Otherwise, only unlabeled samples will train the
        transformer.
    """

    def __init__(self, transformer, estimator, unlabeled=-1,
                 iid=False):
        self.transformer = transformer
        self.estimator = estimator
        self.unlabeled = unlabeled
        self.iid = iid
        self._finished_transformer_fitting = False
        self._finished_estimator_fitting = False

    def fit(self, X, y):
        """Fit the unsupervised transformer and supervised estimator
        
        Parameters
        ----------
        X : array or sparse matrix, shape (n_samples, n_features)
            Feature values

        y : array, sahpe
            Target values, including markers matching `unlabeled`

        Returns
        -------
        self : object
        """
        self._finished_transformer_fitting = False
        self._finished_estimator_fitting = False
        self._partial_fit(False, X, y)
        return self

    def partial_fit(self, X, y):
        """Partially fit the transformer and/or estimator

        This will use `transformer` and `estimator`'s `fit` or `partial_fit`
        methods (as available) to train the models in an out-of-core fashion.

        Once any labeled samples have been processed, an attempt to further fit
        the transformer model will trigger an error. Similarly, where
        `transformer` or `estimator` does not support `partial_fit`, it may
        only be fit once.
        
        Parameters
        ----------
        X : array or sparse matrix, shape (n_samples, n_features)
            Feature values

        y : array, sahpe
            Target values, including markers matching `unlabeled`

        Returns
        -------
        self : object
        """
        self._partial_fit(True, X, y)
        return self

    def _partial_fit(self, is_partial, X, y):
        if callable(self.unlabeled):
            is_unlabeled = self.unlabeled
        else:
            is_unlabeled = functools.partial(np.equal, self.unlabeled)

        unlabeled_mask = is_unlabeled(y)
        labeled_mask = ~unlabeled_mask
        if self.iid:
            unlabeled_mask = slice(None, None)

        if self.iid or unlabeled_mask.sum():
            if self._finished_transformer_fitting:
                if hasattr(self.transformer, 'partial_fit'):
                    raise RuntimeError('Transfer transformer cannot fit more '
                                       'data: the estimator has been fit.')
                else:
                    raise RuntimeError('Transfer transformer does not support '
                                       'partial_fit.')


            if is_partial and hasattr(self.transformer, 'partial_fit'):
                self.transformer.partial_fit(X[safe_mask(X, unlabeled_mask)])
            else:
                self.transformer.fit(X[safe_mask(X, unlabeled_mask)])
                self._finished_transformer_fitting = True

        if labeled_mask.sum():
            if self._finished_estimator_fitting:
                raise RuntimeError('Transfer estimator does not support '
                                   'partial_fit')

            self._finished_transformer_fitting = True
            Xt = self.transformer.transform(X[safe_mask(X, labeled_mask)])
            y = y[safe_mask(y, labeled_mask)]

            if is_partial and hasattr(self.estimator, 'partial_fit'):
                self.estimator.partial_fit(Xt, y)
            else:
                self.estimator.fit(Xt, y)
                self._finished_estimator_fitting = True

    def predict(self, X):
        """Use the estimator to predict targets for transformed data

        Parameters
        ----------
        X : array or sparse matrix, shape (n_samples, n_features)
            Feature values

        Returns
        -------
        y : array, shape [n_samples]
            Predicted targets
        """
        Xt = self.transformer.transform(X)
        return self.estimator.predict(Xt)

    def predict_proba(self, X):
        """Calculate prediction probability of targets

        Parameters
        ----------
        X : array or sparse matrix, shape (n_samples, n_features)
            Feature values

        Returns
        -------
        y : array, shape (n_samples, n_classes)
            Predicted classification probabilities
        """
        Xt = self.transformer.transform(X)
        return self.estimator.predict_proba(Xt)

    def decision_function(self, X):
        """Calculate prediction decision function

        Parameters
        ----------
        X : array or sparse matrix, shape (n_samples, n_features)
            Feature values

        Returns
        -------
        y : array, shape (n_samples, ...)
            Prediction decision function
        """
        Xt = self.transformer.transform(X)
        return self.estimator.decision_function(Xt)

    def transform(self, X):
        """Use the supervised estimator to transform X

        Parameters
        ----------
        X : array or sparse matrix, shape (n_samples, n_features)
            Feature values

        Returns
        -------
        Xt : array, shape (n_samples, n_out_features)
            Transformed feature values
        """
        Xt = self.transformer.transform(X)
        return self.estimator.transform(Xt)

    def inverse_transform(self, Xt):
        """Reverse both the supervised and unsupervised transformations

        Parameters
        ----------
        Xt : array or sparse matrix, shape (n_samples, n_out_features)
            Transformed feature values

        Returns
        -------
        X : array, shape (n_samples, n_features)
            Feature values
        """
        Xt = self.estimator.inverse_transform(Xt)
        return self.transformer.inverse_transform(Xt)

    def score(self, X, y):
        """Score model predictions against a ground truth

        Parameters
        ----------
        X : array or sparse matrix, shape (n_samples, n_features)
            Feature values
        y : array, shape [n_samples]
            Feature values

        Returns
        -------
        score : float
            Metric of underlying estimator comparing `y` to predicted targets
            for `X`
        """
        Xt = self.transformer.transform(X)
        return self.estimator.score(Xt, y)
