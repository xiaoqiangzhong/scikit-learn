"""
The :mod:`sklearn.metrics.scorer` submodule implements a flexible
interface for model selection and evaluation using
arbitrary score functions.

A Scorer object is a callable that can be passed to
:class:`sklearn.grid_search.GridSearchCV` or
:func:`sklearn.cross_validation.cross_val_score` as the ``scoring`` parameter,
to specify how a model should be evaluated.

The signature of the call is ``(estimator, X, y)`` where ``estimator``
is the model to be evaluated, ``X`` is the test data and ``y`` is the
ground truth labeling (or ``None`` in the case of unsupervised models).
"""

# Authors: Andreas Mueller <amueller@ais.uni-bonn.de>
# Liscence: Simplified BSD

import numpy as np

from . import (r2_score, mean_squared_error, accuracy_score, f1_score,
               auc_score, average_precision_score, precision_score,
               recall_score)

from .cluster import adjusted_rand_score


class Scorer(object):
    """Flexible scores for any estimator.

    This class wraps estimator scoring functions for the use in GridSearchCV
    and cross_val_score. It takes a score function, such as ``accuracy_score``,
    ``mean_squared_error``, ``adjusted_rand_index`` or ``average_precision``
    and provides a call method.

    Parameters
    ----------
    score_func : callable,
        Score function (or loss function) with signature
        ``score_func(y, y_pred, **kwargs)``.

    greater_is_better : boolean, default=score_func.greater_is_better or True
        Whether score_func is a score function (default), meaning high is good,
        or a loss function, meaning low is good.

    needs_threshold : bool, default=score_func.needs_threshold or False
        Whether score_func takes a continuous decision certainty.
        For example ``average_precision`` or the area under the roc curve
        can not be computed using predictions alone, but need the output of
        ``decision_function`` or ``predict_proba``.

    **kwargs : additional arguments
        Additional parameters to be passed to score_func.

    Examples
    --------
    >>> from sklearn.metrics import fbeta_score, Scorer
    >>> ftwo_scorer = Scorer(fbeta_score, beta=2)
    >>> from sklearn.grid_search import GridSearchCV
    >>> from sklearn.svm import LinearSVC
    >>> grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]},
    ...                     scoring=ftwo_scorer)
    """
    def __init__(self, score_func, greater_is_better=None,
                 needs_threshold=None, **kwargs):
        self.score_func = score_func
        self.kwargs = kwargs

        if greater_is_better is None:
            greater_is_better = self._get_annotation(score_func,
                    'greater_is_better', True)
        self.greater_is_better = greater_is_better

        if needs_threshold is None:
            needs_threshold = self._get_annotation(score_func,
                    'needs_threshold', False)
        self.needs_threshold = needs_threshold

    @staticmethod
    def _get_annotation(func, attr, default):
        if hasattr(func, attr):
            return getattr(func, attr)
        while hasattr(func, 'func'):
            func = getattr(func, 'func') # unwrap functools.partial
            if hasattr(func, attr):
                return getattr(func, attr)
        return default

    def __repr__(self):
        kwargs_string = "".join([", %s=%s" % (str(k), str(v))
                                 for k, v in self.kwargs.items()])
        return ("Scorer(score_func=%s, greater_is_better=%s, needs_threshold="
                "%s%s)" % (self.score_func.__name__, self.greater_is_better,
                           self.needs_threshold, kwargs_string))

    def __call__(self, estimator, X, y):
        """Score X and y using the provided estimator.

        Parameters
        ----------
        estimator : object
            Trained estimator to use for scoring.
            If ``needs_threshold`` is True, estimator needs
            to provide ``decision_function`` or ``predict_proba``.
            Otherwise, estimator needs to provide ``predict``.

        X : array-like or sparse matrix
            Test data that will be scored by the estimator.

        y : array-like
            True prediction for X.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        if self.needs_threshold:
            if len(np.unique(y)) > 2:
                raise ValueError("This classification score only "
                                 "supports binary classification.")
            try:
                y_pred = estimator.decision_function(X).ravel()
            except (NotImplementedError, AttributeError):
                y_pred = estimator.predict_proba(X)[:, 1]
            return self.score_func(y, y_pred, **self.kwargs)
        else:
            y_pred = estimator.predict(X)
            return self.score_func(y, y_pred, **self.kwargs)


SCORERS = {name: Scorer(metric)
        for name, metric in [
        # Regression
        ('r2', r2_score),
        ('mse', mean_squared_error),
        # Classification
        ('accuracy', accuracy_score),
        ('f1', f1_score),
        ('precision', precision_score),
        ('recall', recall_score),
        # Classification thresholded
        ('roc_auc', auc_score),
        ('average_precision', average_precision_score),
        # Clustering
        ('ari', adjusted_rand_score),
]}
