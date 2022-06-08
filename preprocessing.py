import pathlib
import win32com.client
import yaml

import nltk
import sklearn.model_selection
import numpy as np
import tensorflow as tf


with open('settings.yaml', 'r') as f:
    params = yaml.safe_load(f)['PARAMETERS']


def import_excel(file: pathlib.Path, password: str, start_cell: tuple[int, int],
                 end_cell: tuple[int, int], sheet: int = 1, cols: list[int] = None):
    xlApp = win32com.client.Dispatch("Excel.Application")

    # https://docs.microsoft.com/en-ca/office/vba/api/Excel.Workbooks
    xlwb = xlApp.Workbooks.Open(file, False, True, None, password)

    # https://docs.microsoft.com/en-ca/office/vba/api/excel.worksheet
    xlws = xlwb.Sheets(sheet)

    return np.array(xlws.Range(xlws.Cells(*start_cell), xlws.Cells(*end_cell)).Value)[:, cols]


def preprocess_text(text_ds: np.ndarray):
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


def make_datasets_from_ndarray(x: np.ndarray, y: np.ndarray, train_per=0.7, val_per=0.15, test_per=0.15):
    assert train_per + val_per + test_per == 1.0

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        x, y, train_size=train_per, stratify=y)
    x_val, x_test, y_val, y_test = sklearn.model_selection.train_test_split(
        x_test, y_test, train_size=val_per / (val_per + test_per), stratify=y_test)

    # Creates datasets from numpy.ndarray
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    train_ds = train_ds.shuffle(params['SHUFFLE_BUFFER_SIZE']).batch(params['BATCH_SIZE'])
    val_ds = val_ds.batch(params['BATCH_SIZE'])
    test_ds = test_ds.batch(params['BATCH_SIZE'])

    return train_ds, val_ds, test_ds


def vectorize_dataset(vectorize_layer: tf.keras.layers.TextVectorization, *args):
    return [text_ds.prefetch(eval(params['AUTOTUNE'])).map(lambda x, y: (vectorize_layer(tf.expand_dims(x, -1)), y)).unbatch() for text_ds in args]


if __name__ == '__main__':
    pass
