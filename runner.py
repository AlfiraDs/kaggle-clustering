import os

from scipy.stats import uniform, randint
from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering, DBSCAN, \
    Birch
from sklearn.decomposition import PCA
from sklearn.utils.fixes import loguniform
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, ParameterSampler
from sklearn.pipeline import Pipeline, FeatureUnion

from ads.utils.metrics import rmse
from ads.utils.feature import get_search_space_fit_features

import consts

data = consts.data
train = consts.train
y = consts.y
test = consts.test

features, fes_steps = get_search_space_fit_features(consts.FEATURES_DIR)

n_jobs = -1


class Pipeline_2(Pipeline):
    def __repr__(self, **kwargs):
        return super().__repr__(N_CHAR_MAX=7000)


fu = FeatureUnion(features, n_jobs=n_jobs, verbose=True)
pipe = Pipeline(
    steps=[
        ('fes', fu),
        ('pca', PCA(n_components=5)),
        # ('model', KMeans(n_clusters=2, n_jobs=n_jobs, random_state=1, tol=1e-4)),
        # ('model', AffinityPropagation(damping=0.5, max_iter=200, convergence_iter=15, copy=True, preference=None, affinity='euclidean', verbose=False)),  # lots of ram
        # ('model', MeanShift(bandwidth=None, seeds=None, bin_seeding=False, min_bin_freq=1, cluster_all=True, n_jobs=n_jobs, max_iter=300)),
        # ('model', SpectralClustering(n_clusters=2, eigen_solver=None, n_components=None, random_state=None, n_init=10, gamma=1.0,
        #                     affinity='rbf', n_neighbors=10, eigen_tol=0.0, assign_labels='kmeans', degree=3, coef0=1, kernel_params=None, n_jobs=n_jobs)),  # lots of ram
        # ('model', AgglomerativeClustering(n_clusters=2, affinity='euclidean', memory='./cash', connectivity=None, compute_full_tree='auto', linkage='ward', distance_threshold=None)),
        # ('model', DBSCAN(eps=0.5, min_samples=5, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=n_jobs)),  # no predict method
        # ('model', Birch(threshold=0.5, branching_factor=50, n_clusters=2, compute_labels=True, copy=True)),  # 0.47
        ('model', LogisticRegression(n_jobs=n_jobs, max_iter=1000))
    ])
param_distributions = {
    # 'model__n_clusters': randint(2, 20),
    # 'model__n_init': randint(10, 100),
    # 'model__tol': loguniform(1e-5, 1e-2),
    # 'model__algorithm': ['auto', 'full', 'elkan'],

}
param_distributions.update(fes_steps)

p = list(ParameterSampler(param_distributions, n_iter=1))


def scoring(model, x, y):
    print(model)


model = RandomizedSearchCV(pipe, param_distributions, n_iter=1, cv=5, scoring='accuracy', n_jobs=n_jobs, verbose=0)
# model = pipe
# model.fit(train.sample(frac=0.1), y=None)

model.fit(train, y)
print(model.best_score_)
print(model.best_estimator_)
# print(model.best_estimator_.labels_)
# print(model.labels_)
