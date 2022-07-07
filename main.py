import pathlib
import yaml

import numpy as np
import tensorflow as tf

from preprocessing import load_ds, normalize_ds
from svm_classifier import train_svm

import joblib

if __name__ == '__main__':
    # hyperparameters from Reddy et al. (2021)
    # epochs = 10
    # v_size = 200
    # dropout = 0.5
    # rec_dropout = 0.4
    # max_seq_length = 1200
    # num_layers = 3
    # num_attrs = 200
    # act_func = nn.Tanh()

    with open('settings.yaml', 'r') as f:
        env_vars = yaml.safe_load(f)

        settings = env_vars['SETTINGS']
        params = env_vars['PARAMETERS']

    vec_ds = load_ds(settings['DATA_DIR'], **params)[0]
    norm_vec_ds = normalize_ds(vec_ds)

    print('Optimizing SVM classifier')
    clf = train_svm(vec_ds, **params)

    joblib.dump(clf, 'models/svc.joblib')
