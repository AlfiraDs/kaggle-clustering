import os
import numpy as np
import pandas as pd
from scipy.stats import uniform, randint
from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering, DBSCAN, \
    Birch
from sklearn.decomposition import PCA
from sklearn.utils.fixes import loguniform
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, ParameterSampler, KFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn import metrics

from metrics import rmse
from utils import get_search_space_fit_features

import consts

data = consts.data
train = consts.train
y = consts.y
test = consts.test

n_jobs = -1
rs = 1
np.random.seed(rs)


def scoring(model, x, y):
    pred = model.predict(x)
    df = pd.DataFrame({
        'l': pred,
        'c': y,
    })
    df['cnt'] = df.groupby(['l', 'c'])['c'].count()
    print(model)


features, fes_steps = get_search_space_fit_features(consts.FEATURES_DIR)
fu = FeatureUnion(features, n_jobs=n_jobs, verbose=0)
model = Pipeline(
    steps=[
        ('fes', fu),
        ('pca', PCA(n_components=5, random_state=rs)),
        # ('model', KMeans(n_clusters=20, n_jobs=n_jobs, random_state=rs, tol=1e-4)),
        # ('model', AffinityPropagation(damping=0.5, max_iter=200, convergence_iter=15, copy=True, preference=None, affinity='euclidean', verbose=False)),  # lots of ram
        # ('model', MeanShift(bandwidth=None, seeds=None, bin_seeding=False, min_bin_freq=1, cluster_all=True, n_jobs=n_jobs, max_iter=300)),
        # ('model', SpectralClustering(n_clusters=2, eigen_solver=None, n_components=None, random_state=None, n_init=10, gamma=1.0,
        #                     affinity='rbf', n_neighbors=10, eigen_tol=0.0, assign_labels='kmeans', degree=3, coef0=1, kernel_params=None, n_jobs=n_jobs)),  # lots of ram
        # ('model', AgglomerativeClustering(n_clusters=2, affinity='euclidean', memory='./cash', connectivity=None, compute_full_tree='auto', linkage='ward', distance_threshold=None)),
        # ('model', DBSCAN(eps=0.5, min_samples=5, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=n_jobs)),  # no predict method
        ('model', Birch(threshold=0.5, branching_factor=50, n_clusters=20, compute_labels=True, copy=True)),  # 0.47
        # ('model', LogisticRegression(n_jobs=n_jobs, max_iter=1000))
    ])
params = list(ParameterSampler(fes_steps, n_iter=1))[0]
model.set_params(**params)
model.fit(data)

for train_index, test_index in KFold().split(train):
    X_train, X_test = train.loc[train_index], train.loc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    pred = model.predict(X_train)
    df = pd.DataFrame({
        'clus': pred,
        'lb': y_train,
    })
    df['cnt'] = df.groupby(['clus', 'lb'], as_index=True)['lb'].transform('count')
    df['cnt_clus'] = df.groupby(['clus'], as_index=True)['lb'].transform('count')
    df['prob'] = df.apply(lambda x: x['cnt'] / x['cnt_clus'] if x['lb'] == 1 else 1 - x['cnt'] / x['cnt_clus'], axis=1)
    p = 0.3
    df['pred'] = df['prob'].apply(lambda prob: np.random.choice([0, 1], p=[1-(prob+p), prob+p]))
    print(metrics.accuracy_score(df['lb'], df['pred']))
    # print(df)



# model.fit(train.sample(frac=0.1), y=None)

# model.fit(train, y)
# print(model.best_score_)
# print(model.best_estimator_)
# print(model.best_estimator_.labels_)
# print(model.labels_)
