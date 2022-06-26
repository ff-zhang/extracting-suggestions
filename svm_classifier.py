import numpy as np
import tensorflow as tf

import sklearn
from sklearn.pipeline import  make_pipeline
from sklearn.model_selection import StratifiedKFold

from joblib import parallel_backend


def train_svm(vec_ds: tf.data.Dataset, **params):
    # x, y = np.empty(0), np.empty(0)
    x, y = [], []

    for x_tensor, y_tensor in vec_ds:
        # x = np.concatenate((x, x_tensor.numpy()))
        # y = np.concatenate((y, y_tensor.numpy()))
        x.append(x_tensor.numpy())
        y.append(y_tensor.numpy())

    x = np.squeeze(np.asarray(x))
    y = np.squeeze(np.asarray(y))

    clf = make_pipeline(
        sklearn.preprocessing.StandardScaler(),
        # sklearn.svm.LinearSVC(class_weight='balanced')
        sklearn.svm.SVC(kernel='rbf', class_weight='balanced')
    )

    skf = StratifiedKFold(n_splits=params['NUM_FOLDS'])
    f1 = []

    for i_train, i_test in skf.split(x, y):
        print(' - Training SVM classifier on test fold')
        with parallel_backend('threading', n_jobs=-1):
            clf.fit(x[i_train], y[i_train])

        print(' - Classifying test fold')
        pred = clf.predict(x[i_test])

        f1.append(sklearn.metrics.f1_score(y[i_test], pred))

        print(sum(y[i_test]), sum(pred))
        print("Precision:", sklearn.metrics.precision_score(y[i_test], pred))
        print("Recall:", sklearn.metrics.recall_score(y[i_test], pred))

    print(f'{f1} \nAverage F1 score: {sum(f1) / len(f1)}')


if __name__ == '__main__':
    pass
