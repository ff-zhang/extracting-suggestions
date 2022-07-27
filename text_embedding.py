import math
import pathlib

import tensorflow as tf

from gensim.models import KeyedVectors

import tqdm
import io


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

    def save(self, vectorize_layer: tf.keras.layers.TextVectorization):
        weights = self.get_layer('w2v_embedding').get_weights()[0]
        vocab = vectorize_layer.get_vocabulary()

        out_v = io.open('models/vectors.tsv', 'w', encoding='utf-8')
        out_m = io.open('models/metadata.tsv', 'w', encoding='utf-8')

        for index, word in enumerate(vocab):
            if index == 0:
                continue  # skip 0, it's padding.

            vec = weights[index]
            out_v.write('\t'.join([str(x) for x in vec]) + "\n")
            out_m.write(word + "\n")

        out_v.close()
        out_m.close()


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


def load_word2vec(model_dir: pathlib.Path, model_file: str = 'word2vec-google-news-300.gz', **params):
    """Get a Keras 'Embedding' layer with weights set from Word2Vec model's learned word embeddings.

    Returns
    -------
    `keras.layers.Embedding`
        Embedding layer, to be used as input to deeper network layers.

    """
    word2vec_vectors = KeyedVectors.load_word2vec_format((model_dir / model_file).as_posix(), binary=True, limit=params['VOCAB_LIMIT'])
    weights = word2vec_vectors.vectors  # vectors themselves, a 2D numpy array
    index_to_key = word2vec_vectors.index_to_key  # which row in `weights` corresponds to which word

    layer = tf.keras.layers.Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        # weights=[weights],
        trainable=False,
        name='word2vec'
    )

    return layer, weights


if __name__ == '__main__':
    from preprocessing import load_env_vars, load_vec_ds

    settings, params = load_env_vars()

    vectorize_layer, vec_ds_list = load_vec_ds(settings['BASE_DIR'] / settings['DATA_DIR'],
                                               get_layer=True, is_xlsx=False, **params)

    word2vec, weights = load_word2vec(settings['BASE_DIR'] / settings['MODEL_DIR'], **params)
