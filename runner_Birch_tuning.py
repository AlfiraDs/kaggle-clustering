import os
import numpy as np
import pandas as pd
from scipy.stats import uniform, randint
from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering, DBSCAN, \
    Birch
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
# from sklearn.utils.fixes import loguniform
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, ParameterSampler, KFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn import metrics

from ads.utils.metrics import rmse
from ads.utils.feature import get_search_space_fit_features

import consts

data = consts.data
train = consts.train
y = consts.y
test = consts.test

n_jobs = 1
rs = 1
np.random.seed(rs)


def scoring(model, x, y):
    pred = model.predict(x) * -1 + 1
    score = metrics.accuracy_score(y, pred)
    return score



features, fes_steps = get_search_space_fit_features(consts.FEATURES_DIR)
fu = FeatureUnion(features, n_jobs=n_jobs)
model = Pipeline(
    steps=[
        ('fes', fu),
        ('pca', PCA(n_components=5, random_state=rs)),
        ('model', KMeans(n_clusters=2, n_jobs=n_jobs, random_state=rs, tol=1e-4)),
        # ('model', AffinityPropagation(damping=0.5, max_iter=200, convergence_iter=15, copy=True, preference=None, affinity='euclidean', verbose=False)),  # lots of ram
        # ('model', MeanShift(bandwidth=None, seeds=None, bin_seeding=False, min_bin_freq=1, cluster_all=True, n_jobs=n_jobs, max_iter=300)),
        # ('model', SpectralClustering(n_clusters=2, eigen_solver=None, n_components=None, random_state=None, n_init=10, gamma=1.0,
        #                     affinity='rbf', n_neighbors=10, eigen_tol=0.0, assign_labels='kmeans', degree=3, coef0=1, kernel_params=None, n_jobs=n_jobs)),  # lots of ram
        # ('model', AgglomerativeClustering(n_clusters=2, affinity='euclidean', memory='./cash', connectivity=None, compute_full_tree='auto', linkage='ward', distance_threshold=None)),
        # ('model', DBSCAN(eps=0.5, min_samples=5, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=n_jobs)),  # no predict method
        # ('model', Birch(threshold=0.5, branching_factor=50, n_clusters=2, compute_labels=True, copy=True)),  # 0.47
        # ('model', GaussianMixture(n_components=2, covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=300, n_init=1,
        #                           init_params='kmeans', weights_init=None, means_init=None, precisions_init=None, random_state=rs,
        #                           warm_start=False, verbose=0, verbose_interval=10)),
        # ('model', LogisticRegression(n_jobs=n_jobs, max_iter=1000))
    ])
param_distributions = {
    # 'model__threshold': uniform(0, 1),
    # 'model__branching_factor': randint(1, 1000),
    # 'model__threshold': loguniform(1e-5, 1e-2),
    # 'model__algorithm': ['auto', 'full', 'elkan'],
    # 'pca__n_components': randint(1, 15),
    # 'model__tol': loguniform(1e-5, 1e-2),
    # 'model__reg_covar': loguniform(1e-7, 1e-3),

}
param_distributions.update(fes_steps)
# model = RandomizedSearchCV(model, param_distributions, n_iter=20, cv=5, scoring=scoring, n_jobs=n_jobs, verbose=1)
params = list(ParameterSampler(fes_steps, n_iter=1))[0]
model.set_params(**params)

# model.fit(train, y)
# print(model.best_score_)
# print(model.best_estimator_)
model.fit(data)
df = pd.DataFrame({
    'index': test.index,
    'target': model.predict(test)
})
df.to_csv(os.path.join(consts.DATA_DIR, 'submission_4.csv'), index=False)
print(metrics.accuracy_score(y, model.predict(train)))
# print(model.best_estimator_.labels_)
# print(model.labels_)
