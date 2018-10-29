import collections
from keras import layers
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


class Docset(object):
    def __init__(self, init_doc):
        self.count = collections.Counter()
        self.add(init_doc)

    def add(self, doc):
        self.count.update(doc)

    def vocabulary(self, num_words):
        counts = [('UNK', -1)] + self.count.most_common(num_words-1)
        return Vocabulary([word for (word, _) in counts ])


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
    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim

    def model(self, vocab):
        """prepare the model"""
        embedding_dim = self.embedding_dim
        in_target, in_context = layers.Input((1, )), layers.Input((1, ))
        embedding = layers.Embedding(vocab.size, embedding_dim, input_length=1, name='embedding')

        target = embedding(in_target)
        target = layers.Reshape((embedding_dim, 1))(target)

        context = embedding(in_context)
        context = layers.Reshape((embedding_dim, 1))(context)

        dot_product = layers.merge.dot([target, context], axes=1)
        dot_product = layers.Reshape((1,))(dot_product)
        output = layers.Dense(1, activation='sigmoid')(dot_product)

        model = models.Model(inputs=[in_target, in_context], outputs=output)
        model.compile(loss='binary_crossentropy', optimizer='rmsprop')

        # for the validation model, apply cosine similarity
        similarity = layers.merge.dot([target, context], axes=1, normalize=True)
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
        return [(ex, (-sim).argsort()[1:top_k+1]) for (ex, sim) in sims]

    def sim(self, target, vocab_size):
        sim = np.zeros((vocab_size,))
        ex_t = np.zeros((1,))
        ex_c = np.zeros((1,))
        for i in range(vocab_size):
            ex_t[0,] = target
            ex_c[0,] = i
            sim[i] = self.sim_model.predict_on_batch([ex_t, ex_c])

        return sim


if __name__ == '__main__':
    logging.basicConfig(format="[%(asctime)s]%(levelname)s:%(name)s:%(message)s",
                        level=logging.INFO)
    logging.info('logging configured')

    filename = 'text8.zip'
    logging.info('Reading corpus')
    training_corpus = str_from_zipfile(filename).split()

    logging.info('Creating vocabulary')
    vocab = Docset(training_corpus).vocabulary(int(1e4))
    logging.info('Encoding corpus')
    tokens = vocab.encode(training_corpus)
    logging.info(list(zip(training_corpus[:7], tokens[:7])))
    del training_corpus

    # Validation set for inspecting training progress
    # Max index of the random sampling from tokens (implies choosing
    # from the the top-N occurring tokens)
    max_token_idx = 100
    valid_samples = np.random.choice(max_token_idx, size=16, replace=False)

    epochs = int(1e6)
    word2vec = SkipGramNegSample(embedding_dim=300)
    model, validation_model = word2vec.model(vocab)
    sim_eval = Similarity(validation_model)

    # Inputs and labels
    sampling_table = seq.make_sampling_table(vocab.size)
    logging.info('Creating skipgrams')
    skipgrams, labels = seq.skipgrams(tokens, vocab.size, window_size=3,
                                      sampling_table=sampling_table)
    # downconvert the target and context vectors int16
    word_target, word_context = zip(*skipgrams)
    word_target = np.array(word_target, dtype='int32')
    word_context = np.array(word_context, dtype='int32')
    logging.info(list(zip(word_target[:10], word_context[:10], labels[:10])))
    del tokens

    # Feeding to the network
    ex_t, ex_c, ex_l = np.zeros((1,)), np.zeros((1,)), np.zeros((1,))
    for cnt in range(epochs):
        idx = np.random.randint(0, len(labels)-1)
        ex_t[0,], ex_c[0,], ex_l[0,] = word_target[idx], word_context[idx], labels[idx]
        loss = model.train_on_batch([ex_t, ex_c], ex_l)
        if cnt % 1000 == 0:
            logging.info("Iteration %s, loss=%s", cnt, loss)
        if cnt % 25000 == 0:
            most_similar = sim_eval.most_similar(valid_samples, vocab)
            log_str = '\n'.join(['Nearest to {}: {}'
                                    .format(vocab.decode([ex]), ', '.join(vocab.decode(sim)))
                                for (ex, sim) in most_similar])
            logging.info(log_str)
