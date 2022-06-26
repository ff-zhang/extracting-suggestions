import math

import numpy as np
import tensorflow as tf
import tqdm


# Based off of Word2Vec model from Google: https://www.tensorflow.org/tutorials/text/word2vec
class Word2Vec(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.target_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=1, name="w2v_embedding")
        self.context_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=4+1)

    def call(self, pair):
        target, context = pair

        # target: (batch, dummy?),  context: (batch, context)
        if len(target.shape) == 2:
            target = tf.squeeze(target, axis=1)

        # target: (batch,)
        word_emb = self.target_embedding(target)
        # word_emb: (batch, embed)
        context_emb = self.context_embedding(context)
        # context_emb: (batch, context, embed)
        dots = tf.einsum('be,bce->bc', word_emb, context_emb)
        # dots: (batch, context)
        return dots


# Generates skip-gram pairs with negative sampling for a list of sequences  (int-encoded sentences)
# based on window size, number of negative samples and vocabulary size.
def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
    # Elements of each training example are appended to these lists.
    targets, contexts, labels = [], [], []

    # Build the sampling table for `vocab_size` tokens.
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

    # Iterate over all sequences (sentences) in the dataset.
    for col_vec in tqdm.tqdm(list(sequences.as_numpy_iterator())):
        sequence = list(tf.reshape(col_vec, math.prod(col_vec.shape)).numpy())

        # Generate positive skip-gram pairs for a sequence (sentence).
        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
              sequence,
              vocabulary_size=vocab_size,
              sampling_table=sampling_table,
              window_size=window_size,
              negative_samples=0)

        # Iterate over each positive skip-gram pair to produce training examples
        # with a positive context word and negative samples.
        for target_word, context_word in positive_skip_grams:
            context_class = tf.expand_dims(tf.constant([context_word], dtype="int64"), 1)
            negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                true_classes=context_class,
                num_true=1,
                num_sampled=num_ns,
                unique=True,
                range_max=vocab_size,
                seed=seed,
                name="negative_sampling")

            # Build context and label vectors (for one target word)
            negative_sampling_candidates = tf.expand_dims(negative_sampling_candidates, 1)

            context = tf.concat([context_class, negative_sampling_candidates], 0)
            label = tf.constant([1] + [0]*num_ns, dtype="int64")

            # Append each element from the training example to global lists.
            targets.append(target_word)
            contexts.append(context)
            labels.append(label)

    return targets, contexts, labels


if __name__ == '__main__':
    import pathlib
    import yaml

    from preprocessing import import_multiple_excel, ds_from_ndarray, create_vectorize_layer, \
    preprocess_text_ds, vectorize_dataset

    with open('settings.yaml', 'r') as f:
        env_vars = yaml.safe_load(f)

        settings = env_vars['SETTINGS']
        params = env_vars['PARAMETERS']

    data_dir = pathlib.Path().resolve() / settings['DATA_DIR']
    files = [data_dir / f for f in ('SALG-Instrument-78901-2.xlsx', 'SALG-Instrument-92396.xlsx')]

    text_ds, code_ds = import_multiple_excel(files, 'matter22', [(1, 1)] * 2, [(89, 15), (353, 18)], [(6, 7), (5, 6)])

    # Only consider entries that have been labelled
    mask = (code_ds != '-') & (code_ds != '+')
    mask_ds = preprocess_text_ds(np.ma.masked_array(text_ds, mask).compressed())
    mask_code = np.ma.masked_array(code_ds, mask).compressed()

    # Creates the training, testing, and validation datasets
    ds_list = ds_from_ndarray(mask_ds, mask_code, **params)
    vectorize_layer = create_vectorize_layer(ds_list)

    # Vectorize the training, validation, and test datasets
    train_vec_ds, val_vec_ds, test_vec_ds = vectorize_dataset(vectorize_layer, *ds_list, **params)

    targets, contexts, labels = generate_training_data(
        sequences=train_vec_ds.map(lambda x, y: x),
        window_size=2,
        num_ns=4,
        vocab_size=vectorize_layer.vocabulary_size(),
        seed=params['SEED']
    )

    targets = np.array(targets)
    contexts = np.array(contexts)[:, :, 0]
    labels = np.array(labels)

    print('\n')
    print(f"targets.shape: {targets.shape}")
    print(f"contexts.shape: {contexts.shape}")
    print(f"labels.shape: {labels.shape}")

    dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
    dataset = dataset.shuffle(params['BUFFER_SIZE']).batch(params['BATCH_SIZE'], drop_remainder=True)

    dataset = dataset.cache().prefetch(buffer_size=eval(params['AUTOTUNE']))

    word2vec = Word2Vec(vectorize_layer.vocabulary_size(), params['EMBEDDING_DIM'])
    word2vec.compile(optimizer='adam',
                     loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'])

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
    word2vec.fit(dataset, epochs=20, callbacks=[tensorboard_callback])

    import io

    weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
    vocab = vectorize_layer.get_vocabulary()

    out_v = io.open('logs/vectors.tsv', 'w', encoding='utf-8')
    out_m = io.open('logs/metadata.tsv', 'w', encoding='utf-8')

    for index, word in enumerate(vocab):
        if index == 0:
            continue  # skip 0, it's padding.

        vec = weights[index]
        out_v.write('\t'.join([str(x) for x in vec]) + "\n")
        out_m.write(word + "\n")

    out_v.close()
    out_m.close()