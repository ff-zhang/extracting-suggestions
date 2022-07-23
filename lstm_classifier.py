import os
import pathlib
import matplotlib.pyplot as plt
from itertools import product

import tensorflow as tf

from epoch_model_checkpoint import EpochModelCheckpoint
from preprocessing import load_env_vars, load_vec_ds
from text_embedding import load_word2vec


def train_rnn(train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, model_dir: pathlib.Path,
              logs_dir: pathlib.Path, hparams: dict, word2vec=None, **params):
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
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
                      ])

    checkpoint_filepath = logs_dir / 'checkpoints' / '-'.join(map(str, hparams.values()))
    checkpoint_filename = 'cp-{epoch:04d}.ckpt'
    if not os.path.exists(checkpoint_filepath):
        os.mkdir(checkpoint_filepath)

    model_checkpoint_callback = EpochModelCheckpoint(
        filepath=checkpoint_filepath / checkpoint_filename,
        frequency=50,
        save_weights_only=True,
        verbose=1
    )

    # Save the weights using the `checkpoint_path` format
    model.save_weights(checkpoint_filepath / checkpoint_filename.format(epoch=0))

    history = model.fit(x=train_ds,
                        epochs=params['EPOCHS'],
                        # verbose=0,
                        validation_data=val_ds,
                        callbacks=[model_checkpoint_callback])

    metrics = model.evaluate(ds_list[2])
    f1 = 0 if metrics[2] * metrics[3] == 0 else (2 * metrics[2] * metrics[3]) / (metrics[2] + metrics[3])
    save_graph(logs_dir / 'graphs' / f'{f1}-{"-".join(map(str, hparams.values()))}.png', history)

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
        'EPSILON': [1e-0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    }

    for comb in product(*hparams.values()):
        hp = {}
        for i, k in enumerate(hparams.keys()):
            hp[k] = comb[i]

        train_rnn(ds_list[0], ds_list[1], model_dir, logs_dir, hp, **params)


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
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    ds_list = [ds.with_options(options) for ds in ds_list]

    print('Training RNN')
    optimize_hyperparameters(ds_list, settings['BASE_DIR'] / settings['MODEL_DIR'],
                             settings['BASE_DIR'] / settings['LOGS_DIR'], **params)
