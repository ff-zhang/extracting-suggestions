import yaml
import pathlib
from typing import Union
import csv

import nltk
import numpy as np

import sklearn.model_selection
import tensorflow as tf


def load_env_vars(file_name: str = 'settings.yaml'):
    cur_dir = pathlib.Path(__file__).parent.resolve()

    with open(cur_dir / file_name, 'r') as f:
        env_vars = yaml.safe_load(f)

        settings = env_vars['SETTINGS']
        settings['BASE_DIR'] = cur_dir

        params = env_vars['PARAMETERS']

    return settings, params


def import_excel(file: pathlib.Path, password: str, start_cell: tuple[int, int],
                 end_cell: tuple[int, int], sheet: int = 1, cols: list[int] = None):
    import win32com.client

    xlApp = win32com.client.Dispatch('Excel.Application')

    # https://docs.microsoft.com/en-ca/office/vba/api/Excel.Workbooks
    xlwb = xlApp.Workbooks.Open(file, False, True, None, password)

    # https://docs.microsoft.com/en-ca/office/vba/api/excel.worksheet
    xlws = xlwb.Sheets(sheet)

    return np.array(xlws.Range(xlws.Cells(*start_cell), xlws.Cells(*end_cell)).Value, dtype='str')[:, cols]


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


def sentence_tokenize(text_ds: np.ndarray):
    sentences = []

    for text in np.nditer(text_ds, flags=['buffered']):
        # Replaces undefined tokens with the empty string
        sentences.extend((sent.replace(u'\uFFFD', '') for sent in nltk.sent_tokenize(str(text))))

    # ensures there are enough entries for np.reshape()
    sentences.extend((text_ds.shape[-1] - len(sentences) % text_ds.shape[-1]) * [''])
    a = np.asarray(sentences).reshape(len(sentences) // text_ds.shape[-1], text_ds.shape[-1])

    with open('text.csv', 'w', newline='') as f:
        csv.writer(f).writerows(a)


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


def normalize_ds(ds: tf.data.Dataset):
    norm = tf.keras.layers.Normalization()
    norm_ds = ds.map(lambda x, y: x)
    norm.adapt(norm_ds)

    return ds.map(lambda x, y: (norm(x), y))


def ds_to_ndarray(vec_ds):
    x, y = [], []

    for x_tensor, y_tensor in vec_ds.unbatch():
        x.extend(x_tensor.numpy())
        y.extend(y_tensor.numpy())

    x = np.squeeze(np.asarray(x))
    y = np.squeeze(np.asarray(y))

    return x, y


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


def vectorize_ds(vectorize_layer: tf.keras.layers.TextVectorization, *args):
    ds_list = []
    # from text_embedding import load_word2vec

    for text_ds in args:
        vec_ds = text_ds.map(lambda x, y: (vectorize_layer(tf.expand_dims(x, -1)), y))
        # word2vec, _ = load_word2vec(settings['BASE_DIR'] / settings['MODEL_DIR'], **params)
        # emb_ds = vec_ds.map(lambda x, y: (word2vec(x), y))
        ds_list.append(vec_ds)

    return ds_list


def load_ds(data_dir: pathlib.Path, is_xlsx: bool = True, prefix: str = '',
            ds_types: tuple[str] = ('train', 'validation', 'test'), **params):
    if is_xlsx:
        files = [data_dir / f for f in ('SALG-Instrument-78901-2.xlsx', 'SALG-Instrument-92396.xlsx')]

        text_ds, code_ds = import_multiple_excel(files, 'matter22', [(1, 1)] * 2, [(89, 15), (353, 18)], [(6, 7), (5, 6)])

        # Only consider entries that have been labelled
        mask = (code_ds != '-') & (code_ds != '+')
        # np.ma.compressed() turns the array into a column vector
        mask_ds = preprocess_text_ds(np.ma.masked_array(text_ds, mask).compressed())
        mask_code = np.ma.masked_array(code_ds, mask).compressed()
        mask_code = np.asarray(list(int(i == '+') for i in mask_code))

        # Creates the training, testing, and validation datasets
        ds_list = ds_from_ndarray(mask_ds, mask_code, **params)

    # loading model saved using tf.data.experimental.load()
    else:
        ds_list = []
        for dir in ds_types:
            ds = tf.data.experimental.load((data_dir / (prefix + dir)).as_posix())
            ds_list.append(ds.batch(params['BATCH_SIZE'], drop_remainder=True))

    return ds_list


def load_vec_ds(ds_dir: pathlib.Path, get_layer: bool = False, is_xlsx: bool = True, **params):
    ds_list = load_ds(ds_dir, is_xlsx=is_xlsx, **params)

    # Vectorize the training, validation, and test datasets
    vectorize_layer = create_vectorize_layer(ds_list, max_tokens=params['MAX_TOKENS'])
    vec_ds_list = vectorize_ds(vectorize_layer, *ds_list)

    if get_layer:
        return vectorize_layer, vec_ds_list
    else:
        return vec_ds_list


if __name__ == '__main__':
    settings, params = load_env_vars()

    data_dir = settings['BASE_DIR'] / settings['DATA_DIR']

    # # Load the dataset in Tensorflow
    train_ds, val_ds, test_ds = load_ds(settings['BASE_DIR'] / '.salg', is_xlsx=True, **params)
    # train_ds, val_ds, test_ds = load_vec_ds(data_dir, is_xlsx=False, **params)

    # Save the dataset to a file
    save_dir = settings['BASE_DIR'] / settings['DATA_DIR']
    tf.data.experimental.save(train_ds, (save_dir / 'sk_train').as_posix())  # as_posix() for Windows
    tf.data.experimental.save(val_ds, (save_dir / 'emb_validation').as_posix())
    tf.data.experimental.save(test_ds, (save_dir / 'sk_test').as_posix())
