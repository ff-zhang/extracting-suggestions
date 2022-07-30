import os
import pathlib
import argparse
from itertools import product

import tensorflow as tf

from preprocessing import load_env_vars, load_vec_ds
from text_embedding import load_word2vec
from epoch_model_checkpoint import EpochModelCheckpoint, save_graph
from f1_score import F1Score


def train_lstm(ds_list: list[tf.data.Dataset], model_dir: pathlib.Path, logs_dir: pathlib.Path,
               hparams: dict, **params):
    print(hparams)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(None,)))
        layer, weights = load_word2vec(model_dir, **params)
        model.add(layer)
        layer.set_weights = [weights]
        for _ in range(hparams['LSTM_LAYERS'] - 1):
            model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(2 * hparams['LSTM_UNITS'], return_sequences=True)))
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hparams['LSTM_UNITS'])))
        model.add(tf.keras.layers.Dense(hparams['DENSE_UNITS'], activation='tanh'))
        model.add(tf.keras.layers.Dropout(hparams['DROPOUT']))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        optimizer = tf.keras.optimizers.Adam(learning_rate=hparams['LEARNING_RATE'], epsilon=hparams['EPSILON'])

        model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                      optimizer=optimizer,
                      metrics=[
                          'accuracy',  # 'precision', 'recall'
                          tf.keras.metrics.Precision(name='precision'),
                          tf.keras.metrics.Recall(name='recall'),
                          F1Score(name='f1'),
                      ])

    checkpoint_filepath = logs_dir / 'checkpoints' / '-'.join(map(str, hparams.values()))
    checkpoint_filename = 'ckpt-{epoch:04d}.ckpt'
    if not os.path.exists(checkpoint_filepath):
        os.mkdir(checkpoint_filepath)

    model_checkpoint_callback = EpochModelCheckpoint(
        checkpoints_dir=checkpoint_filepath,
        file_name=checkpoint_filename,
        frequency=params['CHECKPOINT_FREQ'],
        monitor='val_f1',
        mode='max',
        save_best_only=True,
        num_keep=2,
        save_weights_only=True,
        verbose=0
    )

    # Save the weights using the `checkpoint_path` format
    model.save_weights(checkpoint_filepath / checkpoint_filename.format(epoch=0))

    history = model.fit(x=ds_list[0],
                        epochs=params['EPOCHS'],
                        validation_data=ds_list[1],
                        # callbacks=[model_checkpoint_callback],
                        verbose=0)

    metrics = model.evaluate(ds_list[2])
    f1 = 0 if metrics[2] * metrics[3] == 0 else (2 * metrics[2] * metrics[3]) / (metrics[2] + metrics[3])
    save_graph(logs_dir / 'graphs' / 'lstm' / f'{f1}-{"-".join(map(str, hparams.values()))}.png', history)

    return model, history


def optimize_hyperparameters(ds_list: list[tf.data.Dataset], model_dir: pathlib.Path,
                             logs_dir: pathlib.Path, hparams: dict, **params):
    for comb in product(*hparams.values()):
        hp = {}
        for i, k in enumerate(hparams.keys()):
            hp[k] = comb[i]

        model, _ = train_lstm(ds_list, model_dir, logs_dir, hp, **params)


def get_hyperparameters():
    parser = argparse.ArgumentParser(description='Test hyperparameter combinations.')
    parser.add_argument('--lstm-layers', nargs='*', type=int, default=[1, 2])
    parser.add_argument('--lstm-units', nargs='*', type=int, default=[8, 16, 32, 64])
    parser.add_argument('--dense-units', nargs='*', type=int, default=[4, 8, 16, 32])
    parser.add_argument('--dropout', nargs='*', type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument('-lr', '--learning-rate', nargs='*', type=float, default=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8])
    parser.add_argument('-e', '--epsilon', nargs='*', type=float, default=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8])

    return {k.upper(): v for k, v in vars(parser.parse_args()).items()}


if __name__ == '__main__':
    settings, params = load_env_vars()
    hparams = get_hyperparameters()

    print('Loading vectorized dataset')
    ds_list = load_vec_ds(settings['BASE_DIR'] / settings['DATA_DIR'], is_xlsx=False, **params)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    ds_list = [ds.with_options(options) for ds in ds_list]

    print('Training RNN')
    optimize_hyperparameters(ds_list, settings['BASE_DIR'] / settings['MODEL_DIR'],
                             settings['BASE_DIR'] / settings['LOGS_DIR'], hparams, **params)
