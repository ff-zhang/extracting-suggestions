import os
import pathlib
from operator import itemgetter

import tensorflow as tf
from matplotlib import pyplot as plt


def save_graph(save_path: pathlib.Path, history: tf.keras.callbacks.History):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val loss')
    # plt.plot(history.history['precision'], label='precision')
    # plt.plot(history.history['val_precision'], label='val precision')
    # plt.plot(history.history['recall'], label='recall')
    # plt.plot(history.history['val_recall'], label='val recall')
    plt.plot(history.history['f1'], label='F1')
    plt.plot(history.history['val_f1'], label='val F1')

    plt.title('Training')
    plt.ylabel('Value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper right")

    plt.savefig(save_path)
    plt.close()


def find_checkpoints(model_dir, ascending=True):
    """Get all checkpoints in descending order sorted by (epoch, step).
    The checkpoint names must follow the pattern ckpt-{epoch}-{step}.index.
    """
    checkpoints = tf.io.gfile.glob(os.path.join(model_dir, 'ckpt-*.index'))
    checkpoints = map(lambda s: s.strip('.index'), checkpoints)
    by_step = list()
    for checkpoint in checkpoints:
        checkpoint_name = os.path.basename(checkpoint)
        _, epoch = checkpoint_name.split('-')
        by_step.append((epoch, checkpoint))
    by_step = sorted(by_step, key=itemgetter(0))
    if not ascending:
        by_step = by_step[::-1]
    return list(map(itemgetter(1), by_step))


class EpochModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self,
                 checkpoints_dir,
                 file_name,
                 frequency=1,
                 monitor='val_loss',
                 verbose=0,
                 save_best_only=False,
                 num_keep=0,
                 save_weights_only=False,
                 mode='auto',
                 options=None,
                 **kwargs):
        super(EpochModelCheckpoint, self).__init__(checkpoints_dir / file_name, monitor, verbose,
                                                   save_best_only, save_weights_only, mode, "epoch", options)
        self.epoch = 0
        self.epochs_since_last_save = 0
        self.frequency = frequency
        self.checkpoints_dir = checkpoints_dir
        self.num_keep = num_keep

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        self.epoch = epoch
        if self.epochs_since_last_save % self.frequency == 0:
            self._save_model(epoch=epoch, batch=None, logs=logs)

    def on_train_batch_end(self, batch, logs=None):
        pass

    def on_test_end(self, logs=None):
        logger = tf.get_logger()
        if self.num_keep and self.num_keep > 0:
            checkpoints = find_checkpoints(self.checkpoints_dir, ascending=False)

            to_delete = 0 if self.epoch <= self.frequency * 2 else self.num_keep
            print('\n' + str(self.epoch) + ' ' + str(self.epochs_since_last_save) + ' ' + str(checkpoints) + '\n')
            for checkpoint in checkpoints[to_delete:]:
                logger.debug(f'Removing checkpoint {checkpoint}')
                checkpoint_files = tf.io.gfile.glob(checkpoint + '*')
                for file in checkpoint_files:
                    logger.debug(f'Removing: {file}')
                    tf.io.gfile.remove(file)
