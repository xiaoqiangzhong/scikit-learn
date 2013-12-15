"""Tests for sklearn.semi_supervised.transfer"""

from __future__ import division

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.semi_supervised.transfer import Transfer

from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_array_equal


class MockTransformer(BaseEstimator, TransformerMixin):
    """Adds the number of fit samples to the transformed matrix"""
    def __init__(self):
        self.calls = []

    def fit(self, X, y=None):
        self.calls.append(('fit', X))
        self.n_samples_ = X.shape[0]
        return self

    def transform(self, X):
        return X + self.n_samples_

    def inverse_transform(self, Xt):
        return Xt - self.n_samples_


class MockPartialTransformer(MockTransformer):
    """Adds the number of fit samples to the transformed matrix"""
    def partial_fit(self, X, y=None):
        self.calls.append(('partial_fit', X))
        if hasattr(self, 'n_samples_'):
            self.n_samples_ += X.shape[0]
        else:
            self.n_samples_ = X.shape
        return self


class MockClassifier(BaseEstimator):
    def __init__(self):
        self.calls = []

    def fit(self, X, y):
        self.calls.append(('fit', X, y))
        return self

    def predict(self, X):
        self.calls.append(('predict', X))
        return 5

    def predict_proba(self, X):
        self.calls.append(('predict_proba', X))
        return 5

    def decision_function(self, X):
        self.calls.append(('decision_function', X))
        return 5

    def transform(self, X):
        self.calls.append(('transform', X))
        return 5

    def inverse_transform(self, Xt):
        self.calls.append(('inverse_transform', Xt))
        return 5

    def score(self, X, y):
        self.calls.append(('score', X, y))
        return 5


class MockPartialClassifier(MockClassifier):
    def partial_fit(self, X, y):
        self.calls.append(('partial_fit', X, y))
        return self


X_labeled = np.array([[0],
                       [1],
                       [2],])
y_labeled = np.array([1, 1, 1])

X_unlabeled = np.array([[0],
                         [0],])


X_all = np.vstack([X_labeled, X_unlabeled])
y_all = np.hstack([y_labeled, np.repeat([-1], X_unlabeled.shape[0])])
y_all_nan = np.hstack([y_labeled, np.repeat([np.nan], X_unlabeled.shape[0])])
y_all_hundred = np.hstack([y_labeled, np.repeat([100], X_unlabeled.shape[0])])


def assert_calls_equal(expected, actual):
    assert_equal(len(expected), len(actual))
    for call_expected, call_actual in zip(expected, actual):
        assert_equal(len(call_expected), len(call_actual))
        for arg_expected, arg_actual in zip(call_expected, call_actual):
            assert_array_equal(arg_expected, arg_actual)


def test_transfer_fit():
    transformer = MockTransformer()
    estimator = MockClassifier()
    est = Transfer(transformer, estimator)
    est.fit(X_all, y_all)
    assert_calls_equal([('fit', X_unlabeled)], transformer.calls)
    assert_calls_equal([('fit', X_labeled + len(X_unlabeled), y_labeled)],
                       estimator.calls)


def test_transfer_fit_iid():
    transformer = MockTransformer()
    estimator = MockClassifier()
    est = Transfer(transformer, estimator, iid=True)
    est.fit(X_all, y_all)
    assert_calls_equal([('fit', X_all)], transformer.calls)
    assert_calls_equal([('fit', X_labeled + len(X_all), y_labeled)],
                       estimator.calls)


def test_transfer_fit_nan():
    transformer = MockTransformer()
    estimator = MockClassifier()
    est = Transfer(transformer, estimator, unlabeled=np.isnan)
    est.fit(X_all, y_all_nan)
    assert_calls_equal([('fit', X_unlabeled)], transformer.calls)
    assert_calls_equal([('fit', X_labeled + len(X_unlabeled), y_labeled)],
                       estimator.calls)


def test_transfer_fit_nondefault_unlabeled():
    transformer = MockTransformer()
    estimator = MockClassifier()
    est = Transfer(transformer, estimator, unlabeled=100)
    est.fit(X_all, y_all_hundred)
    assert_calls_equal([('fit', X_unlabeled)], transformer.calls)
    assert_calls_equal([('fit', X_labeled + len(X_unlabeled), y_labeled)],
                       estimator.calls)


def test_transfer_delegation():
    transformer = MockTransformer()
    estimator = MockClassifier()
    est = Transfer(transformer, estimator)
    est.fit(X_all, y_all)

    X = np.array([[1],
                  [5]])
    Xt = X + len(X_unlabeled)

    for attr in ['predict', 'predict_proba', 'decision_function', 'transform']:
        # check hard-coded return value is returned
        assert_equal(5, getattr(est, attr)(X))
        assert_calls_equal([(attr, Xt)], estimator.calls[-1:])

    assert_equal(5, est.score(X, [1, 2]))
    assert_calls_equal([('score', Xt, [1, 2])], estimator.calls[-1:])

    assert_equal(5 - len(X_unlabeled), est.inverse_transform(X))

    assert_equal(5, est.score(X, [1, 2]))
    assert_calls_equal([('score', Xt, [1, 2])], estimator.calls[-1:])


# TODO: check set_params, partial_fit
