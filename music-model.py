#!/usr/bin/env python2.7
import sys

import midi
import numpy as np
import tensorflow as tf


class MusicModel:
    """
    An RNN-based generative model for music.
    """

    def __init__(self, hidden_size, embedding_size, learning_rate, vocab_size):
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.vocab_size = vocab_size

        self.num_layers = 2

        # set up placeholders
        self.inpt = tf.placeholder(tf.int32, [None, None])
        self.output = tf.placeholder(tf.int32, [None, None])
        self.keep_prob = tf.placeholder(tf.float32)

        self.lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, state_is_tuple=True)
        self.cell = tf.nn.rnn_cell.MultiRNNCell([self.lstm] * self.num_layers, state_is_tuple=True)
        self.init_state_0 = tf.placeholder(tf.float32, [None, hidden_size])
        self.init_state_1 = tf.placeholder(tf.float32, [None, hidden_size])
        self.init_state_2 = tf.placeholder(tf.float32, [None, hidden_size])
        self.init_state_3 = tf.placeholder(tf.float32, [None, hidden_size])
        firstTuple = tf.nn.rnn_cell.LSTMStateTuple(self.init_state_0, self.init_state_1)
        secondTuple = tf.nn.rnn_cell.LSTMStateTuple(self.init_state_2, self.init_state_3)
        self.init_state = (firstTuple, secondTuple)

        # initialize weight variables
        E = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0))
        W1 = tf.Variable(tf.truncated_normal([hidden_size, vocab_size], stddev=0.1))
        B1 = tf.Variable(tf.random_uniform([vocab_size], -1.0, 1.0))

        # build computation graph
        embd = tf.nn.embedding_lookup(E, self.inpt)
        new_embeddings = tf.nn.dropout(embd, self.keep_prob)
        outputs, state = tf.nn.dynamic_rnn(self.cell, new_embeddings, initial_state=self.init_state)
        self.firstState = state[0][0]
        self.secondState = state[0][1]
        self.thirdState = state[1][0]
        self.fourthState = state[1][1]
        dimensions = tf.slice(tf.shape(self.inpt), [0], [1])
        firstDim = tf.reshape(dimensions, [])
        dimensions = tf.slice(tf.shape(self.inpt), [1], [1])
        secondDim = tf.reshape(dimensions, [])
        reshapedOutput = tf.reshape(outputs, [firstDim * secondDim, hidden_size])
        self.logits = tf.matmul(reshapedOutput, W1) + B1

        loss = tf.nn.seq2seq.sequence_loss_by_example([self.logits], [tf.reshape(self.output, [-1])],
                                                      [tf.ones([firstDim * secondDim])])
        loss = tf.reduce_sum(loss)
        self.loss = loss / tf.to_float(firstDim)
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)


def init_weight(shape, name):
    """
    Initialize a Tensor corresponding to a weight matrix with the given shape and name.

    :param shape: Shape of the weight tensor.
    :param name: Name of the weight tensor in the computation graph.
    :return: Tensor object with given shape and name, initialized from a standard normal.
    """
    return tf.Variable(tf.random_normal(shape, stddev=0.1), name=name)


def init_bias(shape, value, name):
    """
    Initialize a Tensor corresponding to a bias vector with the given shape and name.

    :param shape: Shape of the bias vector (as an int, not a list).
    :param value: Value to initialize bias to.
    :param name: Name of the bias vector in the computation graph.
    :return: Tensor (Vector) object with given shape and name, initialized with given bias.
    """
    return tf.Variable(tf.constant(value, shape=[shape]), name=name)


def preprocess_track(track, ids=None):
    """
    Takes a MIDI track, maps them to a series of note tuples, and returns a list of their IDs (along with the mapping
    of note -> ID)
    :param track: A MIDI track
    :param ids: An existing note->ID mapping can be specified with a tuple (vocab, vocab_reverse).
    :return: The track as a sequence of note IDs, a dictionary mapping notes to IDs, and a list where each note's ID
    is its index.
    """
    if ids is None:
        note_vocab = {}
        note_vocab_reverse = []
    else:
        note_vocab, note_vocab_reverse = ids

    note_sequence = []
    prev_event_type = None
    tick = 0
    for event in track:
        event_type = type(event)

        if not issubclass(event_type, midi.NoteEvent):
            continue

        if event_type == prev_event_type:
            # TODO handle nested notes
            # TODO handle rests
            raise ValueError("Two events of type %s in a row" % str(event_type))

        if type(event) == midi.NoteOffEvent and prev_event_type == midi.NoteOnEvent:
            note_tuple = (event.pitch, event.velocity, event.tick)
            note_id = id_from_token(note_tuple, note_vocab, note_vocab_reverse)
            note_sequence.append(note_id)

        prev_event_type = type(event)
        tick += event.tick
    return note_sequence, note_vocab, note_vocab_reverse


def id_from_token(t, vocab, vocab_reverse):
    """
    Returns the ID of a token, given a vocabulary. If the token is not in the vocabulary, it is added and given a
    unique ID.
    :param t: The token for which to get an ID
    :param vocab: A dictionary mapping tokens to IDs. New tokens will be added to this dictionary.
    :param vocab_reverse: A list of tokens, such that each token's index in this list is its ID. New tokens will be
    appended to this list, and their index will be their new ID.
    :return: The ID of the token 't'
    """
    if t in vocab:
        return vocab[t]
    else:
        t_id = len(vocab_reverse)
        vocab[t] = t_id
        vocab_reverse.append(t)
        return t_id


# TODO training data could be a list of tracks, so we can train on multiple files
def train_model(sess, model, train_data, num_epochs, batch_size, num_steps):
    """
    Trains a music model on the given data, within the given tensorflow session.
    :param sess: The tensorflow session to use
    :param model: The given model to train
    :param train_data: Data on which to train, as a sequence of note IDs
    :param num_epochs: The number of epochs to run on the training data
    :param batch_size: The batch size
    :param num_steps: The number of steps to unroll the RNN in training
    """
    for i in range(num_epochs):
        print ("Epoch %d!!!" % (i + 1))
        total_error = 0.0
        x = 0
        count = 0
        state1 = sess.run(model.lstm.zero_state(batch_size, tf.float32))
        tmp = list()
        tmp.append(state1[0])
        tmp.append(state1[1])
        state1 = tmp
        state2 = sess.run(model.lstm.zero_state(batch_size, tf.float32))
        tmp2 = list()
        tmp2.append(state2[0])
        tmp2.append(state2[1])
        state2 = tmp2
        while (x + batch_size * num_steps) < len(train_data):
            count += num_steps
            inputs = train_data[x:x + batch_size * num_steps]
            inputs = np.reshape(inputs, [batch_size, num_steps])
            outputs = train_data[x + 1:x + batch_size * num_steps + 1]
            outputs = np.reshape(outputs, [batch_size, num_steps])
            x += batch_size * num_steps
            feed = {model.inpt: inputs, model.output: outputs, model.keep_prob: 0.5, model.init_state_0: state1[0],
                    model.init_state_1: state1[1],
                    model.init_state_2: state2[0], model.init_state_3: state2[1]}

            err, state1[0], state1[1], state2[0], state2[1], probabilities, _ = sess.run(
                [model.loss, model.firstState, model.secondState, model.thirdState, model.fourthState, model.logits,
                 model.train_step], feed)
            total_error += err
            print np.exp(total_error / count)


# TODO this shares a lot of code with training, we might be able to abstract some of this out
# TODO we shouldn't use train_data in here
def generate_music(sess, model, num_notes, train_data):
    inputs = train_data[0:len(train_data) - 1]
    outputs = train_data[1:len(train_data)]
    batch_size = 1
    num_steps = 1
    x = 0
    count = 0
    total_error = 0.0
    state1 = sess.run(model.lstm.zero_state(batch_size, tf.float32))
    tmp = list()
    tmp.append(state1[0])
    tmp.append(state1[1])
    state1 = tmp
    state2 = sess.run(model.lstm.zero_state(batch_size, tf.float32))
    tmp = list()
    tmp.append(state2[0])
    tmp.append(state2[1])
    state2 = tmp
    mostLikelyNotes = list()
    while (x + batch_size * num_steps) < num_notes - 1:
        count += num_steps
        new_inputs = inputs[x:x + batch_size * num_steps]
        new_inputs = np.reshape(new_inputs, [batch_size, num_steps])
        new_outputs = outputs[x + 1:x + batch_size * num_steps + 1]
        new_outputs = np.reshape(new_outputs, [batch_size, num_steps])
        feed = {model.inpt: new_inputs, model.output: new_outputs, model.keep_prob: 1.0, model.init_state_0: state1[0],
                model.init_state_1: state1[1],
                model.init_state_2: state2[0], model.init_state_3: state2[1]}
        err, state1[0], state1[1], state2[0], state2[1], probabilities = sess.run(
            [model.loss, model.firstState, model.secondState, model.thirdState, model.fourthState, model.logits], feed)
        total_error += err
        print np.exp(total_error / count)
        probabilities = np.exp(probabilities)
        probabilities = probabilities[0]
        magnitude = sum(probabilities)
        probabilities = probabilities / magnitude
        print probabilities
        x += batch_size * num_steps
        max_index = np.random.choice(range(len(probabilities)), p=probabilities)
        mostLikelyNotes.append(max_index)

    return mostLikelyNotes


def main():
    # TODO for now we're training on a single song
    train_file_name = sys.argv[1]
    pattern = midi.read_midifile(train_file_name)
    tracks = pattern[1:]
    track0 = tracks[0]
    notes, vocab, vocab_reverse = preprocess_track(track0)
    model = MusicModel(hidden_size=128, embedding_size=128, learning_rate=1e-4, vocab_size=len(vocab))

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    train_model(sess, model, batch_size=5, num_epochs=50, num_steps=5, train_data=notes)
    generated_notes = generate_music(sess, model, num_notes=len(notes), train_data=notes)
    print "Original Notes:"
    print notes
    print "Generated Notes:"
    print generated_notes


if __name__ == '__main__':
    main()
