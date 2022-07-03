import numpy as np
import tensorflow as tf

import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold

import skopt
from skopt.space import Real, Categorical, Integer

from joblib import parallel_backend


def ds_to_ndarray(vec_ds):
    # x, y = np.empty(0), np.empty(0)
    x, y = [], []

    for x_tensor, y_tensor in vec_ds.unbatch():
        # x = np.concatenate((x, x_tensor.numpy()))
        # y = np.concatenate((y, y_tensor.numpy()))
        x.append(x_tensor.numpy())
        y.append(y_tensor.numpy())

    x = np.squeeze(np.asarray(x))
    y = np.squeeze(np.asarray(y))

    return x, y


def train_svm(vec_ds: tf.data.Dataset, **params):
    x, y = ds_to_ndarray(vec_ds)

    clf = make_pipeline(
        sklearn.preprocessing.StandardScaler(),
        # values and model taken from optimize_hyperparameters
        sklearn.svm.SVC(
            C=1708.7949461869523,
            kernel='rbf',
            degree=6,
            gamma=8.514453273694237,
            class_weight='balanced'
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


def optimize_hyperparameters(vec_ds, **params):
    # pipeline class is used as estimator to enable
    # search over different model types
    pipe = sklearn.pipeline.Pipeline([
        ('model', sklearn.svm.SVC())
    ])

    linsvc_search = {
        'model': Categorical([sklearn.svm.LinearSVC(max_iter=100000)]),
        'model__C': Real(1e-6, 1e+6, 'log-uniform'),
    }
    svc_search = {
        'model': Categorical([sklearn.svm.SVC()]),
        'model__C': Real(1e-6, 1e+6, 'log-uniform'),
        'model__gamma': Real(1e-6, 1e+1, 'log-uniform'),
        # 'model__degree': Integer(1, 8), # only impacts polynomial kernels
        # Polynomial kernel excluded as it results in 'ValueError: The dual coefficients or intercepts are not finite.'
        'model__kernel': Categorical(['rbf', 'sigmoid']), # 'poly'
    }

    opt = skopt.BayesSearchCV(
        pipe,
        [(svc_search, 50), (linsvc_search, 50)],
        cv=params['NUM_FOLDS'],
        scoring='f1',
        n_jobs=3,
        verbose=3
    )

    x, y = ds_to_ndarray(vec_ds)

    opt.fit(x, y)

    print("val. score: %s" % opt.best_score_)
    print("best params: %s" % str(opt.best_params_))


if __name__ == '__main__':
    import yaml
    import pathlib

    from preprocessing import import_multiple_excel, create_vectorize_layer, normalize_ds, \
        preprocess_text_ds, vectorize_ds

    with open('settings.yaml', 'r') as f:
        env_vars = yaml.safe_load(f)

        settings = env_vars['SETTINGS']
        params = env_vars['PARAMETERS']

    data_dir = pathlib.Path().resolve() / settings['DATA_DIR']
    files = [data_dir / f for f in ('SALG-Instrument-78901-2.xlsx', 'SALG-Instrument-92396.xlsx')]

    print('Importing raw text data and labeling')
    text_ds, code_ds = import_multiple_excel(files, 'matter22', [(1, 1)] * 2, [(89, 15), (353, 18)], [(6, 7), (5, 6)])

    print('Converting data into a numpy array')
    # Only consider entries that have been labelled
    mask = (code_ds != '-') & (code_ds != '+')
    mask_ds = preprocess_text_ds(np.ma.masked_array(text_ds, mask).compressed())
    mask_code = np.ma.masked_array(code_ds, mask).compressed()
    mask_code = np.asarray(list(int(i == '+') for i in mask_code))

    print('Converting data into vectorized Tensorflow dataset')
    # Creates the dataset in Tensorflow
    ds = tf.data.Dataset.from_tensor_slices((mask_ds, mask_code)).batch(params['BATCH_SIZE'], drop_remainder=True)
    vectorize_layer = create_vectorize_layer(ds)

    # Vectorize the dataset
    vec_ds = vectorize_ds(vectorize_layer, ds, **params)[0]
    norm_vec_ds = normalize_ds(vec_ds)

    print('Optimizing SVM classifier')
    # train_svm(vec_ds, **params)
    optimize_hyperparameters(vec_ds, **params)
