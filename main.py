from preprocessing import load_env_vars, load_vec_ds, normalize_ds
from svm_classifier import train_svm

import joblib

if __name__ == '__main__':
    settings, params = load_env_vars()

    vec_ds = load_vec_ds(settings['DATA_DIR'], **params)[0]
    norm_vec_ds = normalize_ds(vec_ds)

    print('Optimizing SVM classifier')
    clf = train_svm(vec_ds, **params)

    joblib.dump(clf, 'models/svc.joblib')
