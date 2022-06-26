import pathlib
import yaml

import numpy as np
import tensorflow as tf

from preprocessing import ds_from_ndarray, import_multiple_excel, create_vectorize_layer, \
    kfold_ds_from_ndarray, preprocess_text_ds, vectorize_dataset
from svm_classifier import train_svm
if __name__ == '__main__':
    # hyperparameters from Reddy et al. (2021)
    # epochs = 10
    # v_size = 200
    # dropout = 0.5
    # rec_dropout = 0.4
    # max_seq_length = 1200
    # num_layers = 3
    # num_attrs = 200
    # act_func = nn.Tanh()

    with open('settings.yaml', 'r') as f:
        env_vars = yaml.safe_load(f)

        settings = env_vars['SETTINGS']
        params = env_vars['PARAMETERS']

    data_dir = pathlib.Path().resolve() / settings['DATA_DIR']
    files = [data_dir / f for f in ('SALG-Instrument-78901-2.xlsx', 'SALG-Instrument-92396.xlsx')]

    print('Importing raw text data and labeling')
    text_ds, code_ds = import_multiple_excel(files, 'matter22', [(1, 1)] * 2, [(89, 15), (353, 18)], [(6, 7), (5, 6)])

    print('Converting data into a numpy array')
    # Only consider entries that have been labelled
    mask = (code_ds != '-') & (code_ds != '+')
    mask_ds = preprocess_text_ds(np.ma.masked_array(text_ds, mask).compressed())
    mask_code = np.ma.masked_array(code_ds, mask).compressed()
    mask_code = np.asarray(list(int(i == '+') for i in mask_code))

    print('Converting data into vectorized Tensorflow dataset')
    # Creates the dataset in Tensorflow
    ds = tf.data.Dataset.from_tensor_slices((mask_ds, mask_code)).batch(params['BATCH_SIZE'], drop_remainder=True)
    vectorize_layer = create_vectorize_layer(ds)

    # Vectorize the dataset
    vec_ds = vectorize_dataset(vectorize_layer, ds, **params)[0]

    print('Training SVM classifier')
    train_svm(vec_ds, **params)

