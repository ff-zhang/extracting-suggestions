import pathlib
from typing import Union

import win32com.client
import yaml

import sklearn.model_selection

import nltk
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


def import_multiple_excel(files: list[pathlib.Path], password: str, starts: list[tuple],
                          ends: list[tuple], sheets: list[tuple], cols: list[Union[list, None]] = None):
    assert len(files) == len(starts) == len(ends) == len(sheets)
    if cols:
        assert len(files) == len(cols)

    text_ds, code_ds = np.empty(0), np.empty(0)

    for i in range(len(files)):
        if cols:
            f_text_ds = import_excel(files[i], password, starts[i], ends[i], sheets[i][0], cols[i])
            f_code_ds = import_excel(files[i], password, starts[i], ends[i], sheets[i][1], cols[i])

        else:
            f_text_ds = import_excel(files[i], password, starts[i], ends[i], sheets[i][0])
            f_code_ds = import_excel(files[i], password, starts[i], ends[i], sheets[i][1])

        text_ds = np.concatenate((text_ds, f_text_ds.flatten()))
        code_ds = np.concatenate((code_ds, f_code_ds.flatten()))

    return text_ds, code_ds


def preprocess_text_ds(text_ds: np.ndarray):
    processed, len_max = [], 0

    lemmatizer = nltk.stem.WordNetLemmatizer()

    for text in text_ds:
        sentence = text.lower()
        words = nltk.tokenize.word_tokenize(sentence)

        processed.append(np.array([lemmatizer.lemmatize(w) for w in words]))
        len_max = len(words) if len_max < len(words) else len_max

    processed = np.array(processed, dtype=object)

    # Note this fills empty row with 7.748604185489348e-304 or 1.3886565059157e-311
    out = np.asarray([np.pad(a, (0, len_max - len(a)), 'empty') if len(a) else np.empty(
        len_max, dtype=str) for a in processed], dtype=str)

    return out


def ds_from_ndarray(x: np.ndarray, y: np.ndarray, train_per=0.7, val_per=0.15, test_per=0.15, **params):
    assert train_per + val_per + test_per == 1.0
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


def kfold_ds_from_ndarray(x: np.ndarray, y: np.ndarray, num_split: int):
    def _gen():
        for i_train, i_test in sklearn.model_selection.KFold(num_split).split(x):
            x_train, x_test = x[i_train], y[i_test]
            y_train, y_test = y[i_train], y[i_test]
            yield x_train, y_train, x_test, y_test

    return tf.data.Dataset.from_generator(_gen, output_signature=())


def create_vectorize_layer(ds_list: tuple[tf.data.Dataset], max_tokens: int = None, **params):
    # Text vectorization layer
    vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens=max_tokens,
        standardize=None,
        split=None,
        output_mode='int',
        output_sequence_length=None,
    )

    # ds = tf.data.Dataset.zip(ds_list).flat_map(lambda *args: _concatenate_ds(args))

    ds_text_list = [ds.map(lambda x, y: x).unbatch() for ds in ds_list]
    choice_ds = [tf.cast(0, tf.int64)] * ds_text_list[0].cardinality().numpy()
    for i, ds in enumerate(ds_text_list[1:]):
        # Casting required as the default is tf.int32 and tf.int64 is required
        choice_ds.extend([tf.cast(i + 1, tf.int64)] * ds.cardinality().numpy())

    choice_ds = tf.data.Dataset.from_tensor_slices(choice_ds)

    # Make a text-only dataset (without labels), then call adapt
    vectorize_layer.adapt(tf.data.Dataset.choose_from_datasets(ds_text_list, choice_ds))

    return vectorize_layer


def vectorize_ds(vectorize_layer: tf.keras.layers.TextVectorization, *args, **params):
    return [text_ds.map(lambda x, y: (vectorize_layer(tf.expand_dims(x, -1)), y)) for text_ds in args]


def load_ds(ds_dir: str, get_layer: bool = False, **params):
    data_dir = pathlib.Path().resolve() / ds_dir
    files = [data_dir / f for f in ('SALG-Instrument-78901-2.xlsx', 'SALG-Instrument-92396.xlsx')]

    text_ds, code_ds = import_multiple_excel(files, 'matter22', [(1, 1)] * 2, [(89, 15), (353, 18)], [(6, 7), (5, 6)])

    # Only consider entries that have been labelled
    mask = (code_ds != '-') & (code_ds != '+')
    mask_ds = preprocess_text_ds(np.ma.masked_array(text_ds, mask).compressed())
    mask_code = np.ma.masked_array(code_ds, mask).compressed()
    mask_code = np.asarray(list(int(i == '+') for i in mask_code))

    # Creates the training, testing, and validation datasets
    ds_list = ds_from_ndarray(mask_ds, mask_code, **params)

    # Vectorize the training, validation, and test datasets
    vectorize_layer = create_vectorize_layer(ds_list, max_tokens=2000)
    train_vec_ds, val_vec_ds, test_vec_ds = vectorize_ds(vectorize_layer, *ds_list, **params)

    if get_layer:
        return vectorize_layer, (train_vec_ds, val_vec_ds, test_vec_ds)
    else:
        return train_vec_ds, val_vec_ds, test_vec_ds


def normalize_ds(ds: tf.data.Dataset):
    norm = tf.keras.layers.Normalization()
    norm_ds = ds.map(lambda x, y: x)
    norm.adapt(norm_ds)

    return ds.map(lambda x, y: (norm(x), y))


def ds_to_ndarray(vec_ds):
    # x, y = np.empty(0), np.empty(0)
    x, y = [], []

    for x_tensor, y_tensor in vec_ds.unbatch():
        # x = np.concatenate((x, x_tensor.numpy()))
        # y = np.concatenate((y, y_tensor.numpy()))
        x.append(x_tensor.numpy())
        y.append(y_tensor.numpy())

    x = np.squeeze(np.asarray(x))
    y = np.squeeze(np.asarray(y))

    return x, y


if __name__ == '__main__':
    with open('settings.yaml', 'r') as f:
        env_vars = yaml.safe_load(f)

        settings = env_vars['SETTINGS']
        params = env_vars['PARAMETERS']

    data_dir = pathlib.Path().resolve() / settings['DATA_DIR']
    file_name = data_dir / 'SALG-Instrument-78901-2.xlsx'

    array_ds = import_excel(file_name, settings['PASSWORD'], (1, 1), (89, 15), sheet=6).flatten()
    coding = import_excel(file_name, settings['PASSWORD'], (1, 1), (89, 15), sheet=7).flatten()

    # Only consider entries that have been labelled
    mask = (coding != '-') & (coding != '+')
    mask_ds = preprocess_text_ds(np.ma.masked_array(array_ds, mask).compressed())
    mask_code = np.ma.masked_array(coding, mask).compressed()

    print('Converting data into vectorized Tensorflow dataset')
    # Creates the dataset in Tensorflow
    ds = tf.data.Dataset.from_tensor_slices((mask_ds, mask_code)).batch(params['BATCH_SIZE'], drop_remainder=True)
    vectorize_layer = create_vectorize_layer(ds)

    # Vectorize the dataset
    vec_ds = load_ds(settings['DATA_DIR'])
    norm_vec_ds = normalize_ds(ds)
