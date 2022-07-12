import numpy as np
import tensorflow as tf

import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold

import skopt
from skopt.space import Real, Categorical, Integer

from joblib import parallel_backend

from preprocessing import ds_to_ndarray, load_env_vars, load_vec_ds, normalize_ds


def train_svm(vec_ds: tf.data.Dataset, **params):
    x, y = ds_to_ndarray(vec_ds)

    clf = make_pipeline(
        # values and model taken from optimize_hyperparameters
        # sklearn.svm.SVC(
        #     C=1708.7949461869523,
        #     kernel='rbf',
        #     degree=6,
        #     gamma=8.514453273694237,
        #     class_weight='balanced'
        # )
        sklearn.svm.LinearSVC(
            # convergent value: 11.245184667895169
            C=23694.157784623574,
            class_weight='balanced',
            max_iter=1000000000,
            verbose=True
        )
    )

    skf = StratifiedKFold(n_splits=params['NUM_FOLDS'])
    precision, recall, f1 = [], [], []

    for i_train, i_test in skf.split(x, y):
        print(' - Training SVM classifier on test fold')
        with parallel_backend('threading', n_jobs=-1):
            clf.fit(x[i_train], y[i_train])

        print(' - Classifying test fold')
        pred = clf.predict(x[i_test])

        precision.append(sklearn.metrics.precision_score(y[i_test], pred))
        recall.append(sklearn.metrics.recall_score(y[i_test], pred))
        f1.append(sklearn.metrics.f1_score(y[i_test], pred))

        print(sum(y[i_test]), sum(pred))
        # print("Precision:", sklearn.metrics.precision_score(y[i_test], pred))
        # print("Recall:", sklearn.metrics.recall_score(y[i_test], pred))

    print('')
    print(f'{precision} \nAverage precision: {sum(precision) / len(precision)} \n')
    print(f'{recall} \nAverage recall: {sum(recall) / len(recall)} \n')
    print(f'{f1} \nAverage F1 score: {sum(f1) / len(f1)}')

    return clf


def optimize_hyperparameters(vec_ds, **params):
    # pipeline class is used as estimator to enable
    # search over different model types
    pipe = sklearn.pipeline.Pipeline([
        ('model', sklearn.svm.SVC(class_weight='balanced'))
    ])

    linsvc_search = {
        'model': Categorical([sklearn.svm.LinearSVC(max_iter=1000000, verbose=3)]),
        'model__C': Real(1e-6, 1e+18, 'log-uniform'),
    }
    svc_search = {
        'model': Categorical([sklearn.svm.SVC()]),
        'model__C': Real(1e-6, 1e+18, 'log-uniform'),
        'model__gamma': Real(1e-18, 1e+6, 'log-uniform'),
        # 'model__degree': Integer(1, 8),  # only impacts polynomial kernels
        'model__kernel': Categorical(['rbf', 'sigmoid']),  # 'poly'
    }

    opt = skopt.BayesSearchCV(
        pipe,
        [(svc_search, 50), (linsvc_search, 50)],
        n_iter=500,
        scoring='f1',
        n_jobs=6,
        cv=params['NUM_FOLDS'],
        verbose=3
    )

    x, y = ds_to_ndarray(vec_ds)

    opt.fit(x, y)

    print("val. score: %s" % opt.best_score_)
    print("best params: %s" % str(opt.best_params_))


if __name__ == '__main__':
    settings, params = load_env_vars()

    vec_ds = load_vec_ds(settings['DATA_DIR'], **params)[0]
    norm_vec_ds = normalize_ds(vec_ds)

    print('Optimizing SVM classifier')
    train_svm(vec_ds, **params)
    # optimize_hyperparameters(vec_ds, **params)
