
from abc import ABCMeta, abstractmethod
import copy

from .base import clone
from .externals.six import with_metaclass


# TODO: some solution for is_classifier


class BaseEstimatorWrapper(with_metaclass(ABCMeta)):
    """Base class for estimator wappers that delegate get_params and attributes
    """
    @abstractmethod
    def __init__(self, estimator):
        self._setattr('_estimator', estimator)

    def clone(self, *args, **kwargs):
        return type(self)(clone(self._estimator), *args, **kwargs)

    def _setattr(self, attr, value):
        super(BaseEstimatorWrapper, self).__setattr__(attr, value)

    def __getattr__(self, attr):
        return getattr(self._estimator, attr)

    def __setattr__(self, attr, value):
        return setattr(self._estimator, attr, value)


class remember_model(BaseEstimatorWrapper):
    def __init__(self, estimator, memory, ignored_params=[]):
        super(remember_model, self).__init__(estimator)
        self._setattr('_cached_fit', memory.cache(self._cached_fit,
                                                  ignore=['self']))
        self._setattr('_memory', memory)
        self._setattr('_ignored_params', ignored_params)

    def _cached_fit(self, key_params, *args, **kwargs):
        self._estimator.fit(*args, **kwargs)
        return self._estimator.__dict__

    def fit(self, *args, **kwargs):
        ignored = {}
        key_params = self._estimator.get_params(deep=True)
        for param in self._ignored_params:
            try:
                ignored[param] = key_params.pop(param)
            except KeyError:
                pass

        self._estimator.__dict__ = self._cached_fit(key_params, *args,
                                                    **kwargs)
        if ignored:
            self._estimator.set_params(**ignored)
        return self

    def clone(self):
        return super(remember_model, self).clone(self._memory,
                                                 self._ignored_params)

    def fit_transform(self, X, *args, **kwargs):
        self._estimator.transform  # ensure attribute exists
        return self.fit(X, *args, **kwargs).transform(X)


class remember_transform(BaseEstimatorWrapper):
    def __init__(self, estimator, memory):
        super(remember_transform, self).__init__(estimator)
        self._setattr('_memory', memory)
        self._setattr('_transform', memory.cache(self._transform,
                                                 ignore=['self', 'result']))

    def clone(self):
        return super(remember_transform, self).clone(self._memory)

    def fit_transform(self, X, *args, **kwargs):
        if hasattr(self._estimator, 'fit_transform'):
            result = self._estimator.fit_transform(X, *args, **kwargs)
            return self._transform(self._estimator, X, result)
        return self.fit(X, *args, **kwargs).transform(X)

    def transform(self, X):
        return self._transform(self._estimator, X)

    def _transform(self, estimator, X, result=None):
        if result is not None:
            return result
        return estimator.transform(X)


class freeze_model(BaseEstimatorWrapper):
    def clone(self):
        return type(self)(copy.deepcopy(self._estimator))

    def fit(self):
        return self

    def fit_transform(self, X, *args, **kwargs):
        self._estimator.transform  # ensure attribute exists
        return self.fit(X, *args, **kwargs).transform(X)


if __name__ == '__main__':
    from sklearn import svm, pipeline, feature_selection, datasets, grid_search
    from sklearn.externals.joblib import Memory
    p = pipeline.Pipeline([
        ('red', remember_model(feature_selection.SelectKBest(),
                               Memory('/tmp/joblib'),
                               ignored_params=['k'])),
        ('clf', svm.LinearSVC()),
    ])
    iris = datasets.load_iris()

    gs = grid_search.GridSearchCV(p, {
        'red__score_func': [feature_selection.chi2,
                            feature_selection.f_classif],
        'red__k': [1, 2, 3], 'clf__C': [1, .1]})
    gs.fit(iris.data, iris.target)
