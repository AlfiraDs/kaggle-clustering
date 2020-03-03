from ads.utils.feature import NumFeatureDescriptor
from sklearn.base import TransformerMixin, BaseEstimator


class Processor(TransformerMixin, BaseEstimator):
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        ret = x.copy()
        ret.iloc[:, 0] = ret.iloc[:, 0].str.replace('[VE]', '').astype(float)
        return ret


fd = NumFeatureDescriptor('diag_2')
fd.steps = [[(Processor, {})]] + fd.steps
