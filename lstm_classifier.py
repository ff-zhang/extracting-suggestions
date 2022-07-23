import pathlib
import matplotlib.pyplot as plt
from itertools import product

import tensorflow as tf

from preprocessing import load_env_vars, load_vec_ds
from text_embedding import load_word2vec


def train_rnn(train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, model_dir: pathlib.Path,
              hparams: dict, word2vec=None, **params):
    if not word2vec:
        word2vec = load_word2vec(model_dir, **params)

    model = tf.keras.Sequential()
    model.add(word2vec)
    print(hparams)
    for _ in range(hparams['LSTM_LAYERS'] - 1):
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(2 * hparams['LSTM_UNITS'], return_sequences=True)))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hparams['LSTM_UNITS'])))
    model.add(tf.keras.layers.Dense(hparams['DENSE_UNITS'], activation='tanh'))
    model.add(tf.keras.layers.Dropout(hparams['DROPOUT']))
    model.add(tf.keras.layers.Dense(1, activation=hparams['ACTIVATION']))

    model.summary()

    if hparams['OPTIMIZER'] == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=hparams['LEARNING_RATE'], epsilon=hparams['EPSILON'])
    elif hparams['OPTIMIZER'] == 'adagrad':
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=hparams['LEARNING_RATE'], epsilon=hparams['EPSILON'])

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=optimizer,
                  metrics=[
                      'accuracy',  # 'precision', 'recall'
                      tf.keras.metrics.Precision(name='precision'),
                      tf.keras.metrics.Recall(name='recall'),
                      # tfa.metrics.F1Score(num_classes=1)
                  ])

    history = model.fit(train_ds, epochs=hparams['EPOCHS'], validation_data=val_ds, verbose=0)

    return model, history


def optimize_hyperparameters(ds_list: list[tf.data.Dataset], model_dir: pathlib.Path,
              logs_dir: pathlib.Path, **params):
    hparams = {
        'LSTM_LAYERS': [1, 2],
        'LSTM_UNITS': [16, 32, 64],
        'DENSE_UNITS': [4, 8, 16, 32, 64],
        'DROPOUT': [0.1, 0.2, 0.3, 0.4, 0.5],
        'ACTIVATION': ['sigmoid'],
        'OPTIMIZER': ['adam'],
        'LEARNING_RATE': [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
        'EPSILON': [1e-0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
        'EPOCHS': [1000],
    }

    w2v = load_word2vec(model_dir, **params)

    for comb in product(*hparams.values()):
        hp = {}
        for i, k in enumerate(hparams.keys()):
            hp[k] = comb[i]

        model, history = train_rnn(ds_list[0], ds_list[1], model_dir, hp, word2vec=w2v, **params)
        metrics = model.evaluate(ds_list[2])
        f1 = 0 if metrics[2] * metrics[3] == 0 else (2 * metrics[2] * metrics[3]) / (metrics[2] + metrics[3])
        save_graph(logs_dir / 'graphs' / f'{f1}-{"-".join(map(str, hp.values()))}.png', history)


def save_graph(save_path: pathlib.Path, history: tf.keras.callbacks.History):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.plot(history.history['precision'], label='precision')
    plt.plot(history.history['val_precision'], label='val precision')
    plt.plot(history.history['recall'], label='recall')
    plt.plot(history.history['val_recall'], label='val recall')

    plt.title('Training')
    plt.ylabel('Value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")

    plt.savefig(save_path)
    plt.close()


if __name__ == '__main__':
    settings, params = load_env_vars()

    print('Loading vectorized dataset')
    ds_list = load_vec_ds(settings['BASE_DIR'] / settings['DATA_DIR'], is_xlsx=False, **params)

    print('Training RNN')
    optimize_hyperparameters(ds_list, settings['BASE_DIR'] / settings['MODEL_DIR'],
                             settings['BASE_DIR'] / settings['LOGS_DIR'], **params)
