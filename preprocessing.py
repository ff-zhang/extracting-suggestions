import pathlib
import win32com.client

import nltk
import sklearn.model_selection
import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 128


def import_excel(file: pathlib.Path, password: str, start_cell: tuple[int, int],
                           end_cell: tuple[int, int], sheet: int = 1, cols: list[int] = None):
    xlApp = win32com.client.Dispatch("Excel.Application")

    # https://docs.microsoft.com/en-ca/office/vba/api/Excel.Workbooks
    xlwb = xlApp.Workbooks.Open(file, False, True, None, password)

    # https://docs.microsoft.com/en-ca/office/vba/api/excel.worksheet
    xlws = xlwb.Sheets(sheet)

    return np.array(xlws.Range(xlws.Cells(*start_cell), xlws.Cells(*end_cell)).Value)[:, cols]


def preprocess(text_ds: np.ndarray):
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


def make_datasets_from_df(x: np.ndarray, y: np.ndarray, train_size=0.7, val_size=0.15):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        x, y, train_size=train_size, stratify=y)

    # Reserve val_size percentage of training samples for validation
    val_num = int(x_train.shape[0] * val_size)
    x_val = x_train[-val_num:]
    y_val = y_train[-val_num:]
    x_train = x_train[:-val_num]
    y_train = y_train[:-val_num]

    # Creates datasets from numpy.ndarray
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    train_ds = train_ds.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    val_ds = val_ds.batch(BATCH_SIZE)
    test_ds = test_ds.batch(BATCH_SIZE)

    return train_ds, val_ds, test_ds


if __name__ == '__main__':
    import yaml

    with open('settings.yaml', 'r') as f:
        settings = yaml.safe_load(f)['SETTINGS']

    data_dir = pathlib.Path().resolve()/settings['DATA_DIR']
    file_name = data_dir/settings['EXCEL_FILE']

    # Columns containing free response questions
    cols = [14, 15, 27, 35, 40, 44, 45, 54, 55, 70, 76, 80, 92, 93, 94]

    dataset = import_excel(file_name, settings['PASSWORD'], (2, 3), (351, 97), sheet=3, cols=cols)
    # Turns array of data into column vector
    dataset = dataset.flatten()[..., None]
    dataset = preprocess(dataset)

    coding = import_excel(file_name, settings['PASSWORD'], (2, 2), (351, 16), sheet=4).flatten()
    # Temporary (simple) coding and will be removed
    for i in range(coding.shape[0]):
        coding[i] = bool(coding[i])

    # Creates the training, testing, and validation datasets
    train_ds, val_ds, test_ds = make_datasets_from_df(dataset, coding)

    # Text vectorization layer
    vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens=None,
        standardize=None,
        split=None,
        output_mode='int',
        output_sequence_length=None,
    )

    # Make a text-only dataset (without labels), then call adapt
    train_text = train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(train_text)

    # Retrieve a batch of reviews and labels from the dataset
    text_batch, label_batch = next(iter(train_ds))
    first_review, first_label = text_batch[0], label_batch[0]

    # Vectorize the training, validation, and test datasets
    train_ds = train_ds.map(lambda text, label: (vectorize_layer(tf.expand_dims(text, -1)), label))
    val_ds = val_ds.map(lambda text, label: (vectorize_layer(tf.expand_dims(text, -1)), label))
    test_ds = test_ds.map(lambda text, label: (vectorize_layer(tf.expand_dims(text, -1)), label))
