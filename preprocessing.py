import pathlib
import win32com.client
import yaml

import nltk
import sklearn.model_selection
import numpy as np
import tensorflow as tf


def import_excel(file: pathlib.Path, password: str, start_cell: tuple[int, int],
                 end_cell: tuple[int, int], sheet: int = 1, cols: list[int] = None):
    xlApp = win32com.client.Dispatch("Excel.Application")

    # https://docs.microsoft.com/en-ca/office/vba/api/Excel.Workbooks
    xlwb = xlApp.Workbooks.Open(file, False, True, None, password)

    # https://docs.microsoft.com/en-ca/office/vba/api/excel.worksheet
    xlws = xlwb.Sheets(sheet)

    return np.array(xlws.Range(xlws.Cells(*start_cell), xlws.Cells(*end_cell)).Value)[:, cols]


def preprocess_text_ds(text_ds: np.ndarray):
    processed, len_max = [], 0

    lemmatizer = nltk.stem.WordNetLemmatizer()

    for text in text_ds:
        sentence = text[0].lower()
        words = nltk.tokenize.word_tokenize(sentence)

        processed.append(np.array([lemmatizer.lemmatize(w) for w in words]))
        len_max = len(words) if len_max < len(words) else len_max

    processed = np.array(processed, dtype=object)

    # Note this fills empty row with 7.748604185489348e-304 or 1.3886565059157e-311
    out = np.asarray([np.pad(a, (0, len_max - len(a)), 'empty') if len(a) else np.empty(
        len_max, dtype=str) for a in processed], dtype=str)

    return out


def make_ds_from_ndarray(x: np.ndarray, y: np.ndarray, train_per=0.7, val_per=0.15, test_per=0.15, **params):
    assert train_per + val_per + test_per == 1.0

    x = preprocess_text_ds(x)

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        x, y, train_size=train_per, stratify=y)
    x_val, x_test, y_val, y_test = sklearn.model_selection.train_test_split(
        x_test, y_test, train_size=val_per / (val_per + test_per), stratify=y_test)

    # Creates datasets from numpy.ndarray
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    train_ds = train_ds.shuffle(params['SHUFFLE_BUFFER_SIZE']).batch(params['BATCH_SIZE'], drop_remainder=True)
    val_ds = val_ds.batch(params['BATCH_SIZE'], drop_remainder=True)
    test_ds = test_ds.batch(params['BATCH_SIZE'], drop_remainder=True)

    return train_ds, val_ds, test_ds


def create_vectorize_layer(ds_list: tuple[tf.data.Dataset], **params):
    # Combines a list of datasets into one
    def _concatenate_ds(ds_list: tuple[tf.data.Dataset]):
        ds0 = tf.data.Dataset.from_tensors(ds_list[0])
        for ds1 in ds_list[1:]:
            ds0 = ds0.concatenate(tf.data.Dataset.from_tensors(ds1))

        return ds0

    # Text vectorization layer
    vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens=None,
        standardize=None,
        split=None,
        output_mode='int',
        output_sequence_length=None,
    )

    # ds = tf.data.Dataset.zip(ds_list).flat_map(lambda *args: _concatenate_ds(args))

    ds_text_list = [ds.map(lambda x, y: x).unbatch() for ds in ds_list]
    choice_ds = [0] * ds_text_list[0].cardinality().numpy()
    for i, ds in enumerate(ds_text_list[1:]):
        # Casting required as the default is tf.int32 and tf.int64 is required
        choice_ds.extend([tf.cast(i + 1, tf.int64)] * ds.cardinality().numpy())
    choice_ds = tf.data.Dataset.from_tensor_slices(choice_ds)

    # Make a text-only dataset (without labels), then call adapt
    vectorize_layer.adapt(tf.data.Dataset.choose_from_datasets(ds_text_list, choice_ds))

    return vectorize_layer


def vectorize_dataset(vectorize_layer: tf.keras.layers.TextVectorization, *args, **params):
    return [text_ds.prefetch(eval(params['AUTOTUNE'])).map(lambda x, y: (vectorize_layer(tf.expand_dims(x, -1)), y)).unbatch() for text_ds in args]


if __name__ == '__main__':
    pass
