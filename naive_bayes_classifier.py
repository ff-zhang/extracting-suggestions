import numpy as np
import tensorflow as tf
import joblib

import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline, make_pipeline

import skopt
from skopt.space import Categorical, Real


from preprocessing import ds_to_ndarray, load_env_vars, load_vec_ds, normalize_ds


def train_nb_classifier(vec_ds: tf.data.Dataset, **params):
    x, y = ds_to_ndarray(vec_ds)

    clf = make_pipeline(
        # MultinomialNB(alpha=0.17601196310151052)
        MultinomialNB(alpha=0.9314472681366592)
    )

    skf = StratifiedKFold(n_splits=params['NUM_FOLDS'])
    precision, recall, f1 = [], [], []

    for i_train, i_test in skf.split(x, y):
        print(' - Training SVM classifier on test fold')
        with joblib.parallel_backend('threading', n_jobs=-1):
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


def optimize_hyperparameters(vec_ds: tf.data.Dataset, **params):
    pipe = Pipeline([
        ('model', sklearn.naive_bayes.GaussianNB())
    ])

    nb_search = {
        'model': Categorical([MultinomialNB(), ComplementNB()]),
        'model__alpha': Real(0, 1, 'uniform')
    }

    opt = skopt.BayesSearchCV(
        pipe,
        nb_search,
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

    return opt


if __name__ == '__main__':
    settings, params = load_env_vars()

    vec_ds = load_vec_ds(settings['BASE_DIR'] / settings['DATA_DIR'], is_xlsx=False, **params)[0]
    norm_vec_ds = normalize_ds(vec_ds)

    print('Optimizing naive Bayes classifier')

    clf = train_nb_classifier(vec_ds, **params)
