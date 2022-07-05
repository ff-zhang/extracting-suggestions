import numpy as np
import tensorflow as tf

import sklearn
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline

import skopt
from skopt.space import Categorical, Real

from preprocessing import ds_to_ndarray

def train_naive_bayes(vec_ds, **params):
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
    import yaml
    import pathlib
    import joblib

    from preprocessing import import_multiple_excel, create_vectorize_layer, \
        normalize_ds, preprocess_text_ds, vectorize_ds

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

    print('Optimizing naive Bayes classifier')
    # train_svm(vec_ds, **params)
    opt = train_naive_bayes(vec_ds, **params)
    joblib.dump(opt, 'models/nbc.joblib')
