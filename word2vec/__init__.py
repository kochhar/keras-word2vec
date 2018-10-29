import collections
from keras import layers
from keras.layers import merge
from keras import models
from keras.preprocessing import sequence as seq
import logging
import numpy as np
import tensorflow as tf
import zipfile


def str_from_zipfile(zipfilename, fileidx=0):
    """Extract the first file enclosed in a zip file as a list of words."""
    with zipfile.ZipFile(zipfilename) as f:
        filename = f.namelist()[fileidx]
        data = tf.compat.as_str(f.read(filename))

    return data


class Vocabulary(object):
    def __init__(self, words):
        self._size = len(words)
        self.word_2_idx = dict([ (word, i) for i, word in enumerate(words) ])
        self.idx_2_word = dict(zip(self.word_2_idx.values(), self.word_2_idx.keys()))

    @property
    def size(self):
        return self._size

    def __str__(self):
        return """Vocabulary(%s words)""" % (len(word_2_idx),)

    def encode(self, doc):
        coder = self.word_2_idx.get
        unk_idx = coder('UNK')
        return [coder(word, unk_idx) for word in doc]

    def decode(self, vec):
        coder = self.idx_2_word.get
        return [coder(idx, 'UNK') for idx in vec]



class SkipGramNegSample(object):
    def __init__(self, vocab_size, embedding_dim):
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

    def skipgrams(self, tokens, window_size=3):
        # Inputs and labels
        sampling_table = seq.make_sampling_table(self.vocab_size)
        skipgrams, labels = seq.skipgrams(tokens, self.vocab_size, window_size=window_size,
                                          sampling_table=sampling_table)
        # downconvert the target and context vectors int16
        word_target, word_context = zip(*skipgrams)
        word_target = np.array(word_target, dtype='int32')
        word_context = np.array(word_context, dtype='int32')
        return (word_target, word_context, labels)

    def model(self):
        """prepare the model"""
        in_target = layers.Input((1, ), name='in_tgt')
        in_context = layers.Input((1, ), name='in_ctx')

        embedding_dim = self.embedding_dim
        embedding = layers.Embedding(self.vocab_size, embedding_dim, input_length=1, name='embedding')

        target = embedding(in_target)
        target = layers.Reshape((embedding_dim, 1), name='target')(target)

        context = embedding(in_context)
        context = layers.Reshape((embedding_dim, 1), name='context')(context)

        dot_product = layers.Dot(axes=1)([target, context])
        dot_product = layers.Reshape((1,), name='dot')(dot_product)
        output = layers.Dense(1, activation='sigmoid', name='output')(dot_product)

        model = models.Model(inputs=[in_target, in_context], outputs=output)
        model.compile(loss='binary_crossentropy', optimizer='rmsprop')

        # for the validation model, apply cosine similarity
        similarity = layers.Dot(axes=1, normalize=True)([target, context])
        similarity = layers.Reshape((1,), name='sim')(similarity)

        validation_model = models.Model(inputs=[in_target, in_context], outputs=similarity)
        return model, validation_model


class Similarity(object):
    def __init__(self, sim_model):
        """Params:
            - sim_model: A keras model which outputs a similarity score given
              two vectors as inputs"""
        self.sim_model = sim_model

    def most_similar(self, examples, vocab, top_k=10):
        sim_fn = self.sim

        sims = [(ex, sim_fn(ex, vocab.size)) for ex in examples]
        return [(ex, (-sim).argsort()[0:top_k+1]) for (ex, sim) in sims]

    def sim(self, target, vocab_size):
        sims = np.zeros((vocab_size,))
        ex_t = np.zeros((1,))
        ex_c = np.zeros((1,))
        for i in range(vocab_size):
            ex_t[0,] = target
            ex_c[0,] = i
            sims[i] = self.sim_model.predict_on_batch([ex_t, ex_c])

        return sims

