import tensorflow as tf

from preprocessing import create_vectorize_layer, load_ds, load_env_vars, load_vec_ds
from text_embedding import load_word2vec


def train_rnn(train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, model_dir: str, **params):
    word2vec = load_word2vec(model_dir, **params)

    model = tf.keras.Sequential([
        # vectorize_layer,
        word2vec,
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation='tanh'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1)
    ])
    model.summary()

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  # metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
                  metrics=['accuracy'])

    model.fit(train_ds, epochs=10, validation_data=val_ds, validation_steps=30)

    return model


if __name__ == '__main__':
    settings, params = load_env_vars()

    print('Loading vectorized dataset')
    ds_list = load_vec_ds(settings['DATA_DIR'], is_xlsx=False, **params)

    print('Training RNN')
    model = train_rnn(ds_list[0], ds_list[1], settings['MODEL_DIR'], **params)

    test_loss, test_acc = model.evaluate(ds_list[2])

    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_acc)
    # print('Test Recall:', test_recall)
    # print('Test Precision:', test_precision)

    model.save('models/rnn')
