from abc import ABCMeta, abstractmethod

from .base import BaseEstimator
from .externals.six import iteritems, with_metaclass


class BaseParameterTranslator(with_metaclass(ABCMeta, BaseEstimator)):

    @abstractmethod
    def _get_translation_rules(self):
        pass

    @abstractmethod
    def _get_inverse_translation_rules(self):
        pass

    @property
    @abstractmethod
    def _subest_param(self):
        pass

    def set_params(self, **params):
        local_params = self.get_params(deep=False)
        translated = {}
        rules = self._get_translation_rules()
        for k, v in iteritems(params):
            if k in local_params:
                translated[k] = v
                continue
            matched = False
            for translate_key, translate_value in rules:
                translations = translate_key(k)
                if translations:
                    matched = True
                    trans_v = translate_value(v)
                    for trans_k in translations:
                        translated[trans_k] = trans_v
            if not matched:
                raise ValueError('Invalid parameter %s for estimator %s' %
                                 (k, self))
        super(BaseParameterTranslator, self).set_params(**translated)

    def get_params(self, deep=True):
        subest_param = self._subest_param
        result = super(BaseParameterTranslator, self).get_params(deep=False)
        if not deep:
            return result
        rules = self._get_inverse_translation_rules()
        for k, v in iteritems(getattr(self, subest_param).get_params(deep)):
            k = subest_param + '__' + k
            for translate_key, translate_value in rules:
                translations = translate_key(k)
                if translations:
                    trans_v = translate_value(v)
                    for trans_k in translations:
                        result[trans_k] = trans_v
        return result


IDENTITY = lambda x: x


class ParameterTie(BaseParameterTranslator):
    def __init__(self, _estimator, _ties):
        self._estimator = _estimator
        self._ties = _ties

    # TODO: handle wildcard
    _subest_param = '_estimator'

    def _get_translation_rules(self):
        return [
            (lambda k: ['_estimator__' + t for t in self._ties.get(k, [])], IDENTITY),
            (lambda k: None if k in self._ties else ['_estimator__' + k], IDENTITY),
        ]

    def _get_inverse_translation_rules(self):
        rule_map = {}
        for alias, targets in iteritems(self._ties):
            for target in targets:
                rule_map['_estimator__' + target] = [alias]
        return [
            (rule_map.get, IDENTITY),
            (lambda k: None if k in rule_map else [k[len('_estimator__'):]], IDENTITY),
        ]


if __name__ == '__main__':
    from .pipeline import FeatureUnion
    from .feature_selection import SelectPercentile
    fu = FeatureUnion([('a', SelectPercentile()), ('b', SelectPercentile())])
    tie = ParameterTie(fu, {'percentile': ['a__percentile', 'b__percentile']})
