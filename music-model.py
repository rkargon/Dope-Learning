#!/usr/bin/env python2.7
import argparse
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
        dimensions = tf.shape(self.inpt)
        first_dim = tf.slice(dimensions, [0], [1])
        first_dim = tf.reshape(first_dim, [])
        second_dim = tf.slice(dimensions, [1], [1])
        second_dim = tf.reshape(second_dim, [])
        reshapedOutput = tf.reshape(outputs, [first_dim * second_dim, hidden_size])
        self.logits = tf.matmul(reshapedOutput, W1) + B1

        loss = tf.nn.seq2seq.sequence_loss_by_example([self.logits], [tf.reshape(self.output, [-1])],
                                                      [tf.ones([first_dim * second_dim])])
        loss = tf.reduce_sum(loss)
        self.loss = loss / tf.to_float(first_dim)
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


def get_notes_from_track(track):
    """
    Converts a single MIDI track to a sequence of note tuples
    :param track: A MIDI track
    :return: A series of tuples (pitch, velocity, duration)
    """
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
            note_sequence.append(note_tuple)

        prev_event_type = type(event)
        tick += event.tick
    return note_sequence


def preprocess_track(track, ids=None):
    """
    Takes a MIDI track, maps them to a series of note tuples, and returns a list of their IDs (along with the mapping
    of note -> ID)
    :param track: A MIDI track
    :param ids: An existing note->ID mapping can be specified with a tuple (vocab, vocab_reverse).
    :return: The track as a sequence of note IDs, a dictionary mapping notes to IDs, and a list where each note's ID
    is its index.
    """
    return build_vocabulary(get_notes_from_track(track), ids=ids)


def build_vocabulary(tokens, ids=None):
    """
    Given a list of tokens, maps them to unique IDs. This is done by keeping a counter, and incrementing it each time
    a new unique token is found.
    :param tokens: A list of objects.
    :param ids: An optional existing vocabulary of IDs specified as a tuple (vocab, vocab_reverse).
    :return: (ids_sequence, vocab, vocab_reverse) The sequence of ids for each object,
    """
    if ids is None:
        vocab = {}
        vocab_reverse = []
    else:
        vocab, vocab_reverse = ids

    ids_sequence = [id_from_token(t, vocab, vocab_reverse) for t in tokens]
    return ids_sequence, vocab, vocab_reverse


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


def train_model(sess, model, train_data, num_epochs, batch_size, num_steps):
    """
    Trains a music model on the given data, within the given tensorflow session.
    :param sess: The tensorflow session to use
    :param model: The given model to train
    :param train_data: A list of tracks where each track is a list of note IDs
    :param num_epochs: The number of epochs to run on the training data
    :param batch_size: The batch size
    :param num_steps: The number of steps to unroll the RNN in training
    """
    for i in range(num_epochs):
        print ("Epoch %d!!!" % (i + 1))
        for track in train_data:
            total_error = 0.0
            x = 0
            # TODO clean up this zest?
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
            while (x + batch_size * num_steps) < len(track):
                inputs = track[x:x + batch_size * num_steps]
                inputs = np.reshape(inputs, [batch_size, num_steps])
                outputs = track[x + 1:x + batch_size * num_steps + 1]
                outputs = np.reshape(outputs, [batch_size, num_steps])
                x += batch_size * num_steps
                feed = {model.inpt: inputs, model.output: outputs, model.keep_prob: 0.5, model.init_state_0: state1[0],
                        model.init_state_1: state1[1],
                        model.init_state_2: state2[0], model.init_state_3: state2[1]}

                err, state1[0], state1[1], state2[0], state2[1], probabilities, _ = sess.run(
                    [model.loss, model.firstState, model.secondState, model.thirdState, model.fourthState, model.logits,
                     model.train_step], feed)
                total_error += err


# TODO this shares a lot of code with training, we might be able to abstract some of this out
# TODO only first note of note_context is used right now
def generate_music(sess, model, num_notes, note_context):
    """
    Uses a trained model to generate notes of mmusic one by one.
    :param sess: The tensorflow session with which to run the model
    :param model: The given music model
    :param num_notes: The number of notes to generate
    :param train_data:
    :return:
    """
    batch_size = 1
    num_steps = 1
    x = 0
    count = 0
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
    previous_note = -1
    max_index = None
    most_likely_notes = list()
    while (x + batch_size * num_steps) < num_notes - 1:
        count += num_steps
        print "****", x
        if previous_note == -1:
            new_inputs = note_context[x:x + batch_size * num_steps]
            new_inputs = np.reshape(new_inputs, [batch_size, num_steps])
        else:
            new_inputs = [max_index]
            new_inputs = np.reshape(new_inputs, [batch_size, num_steps])
        # new_outputs = outputs[x + 1:x + batch_size * num_steps + 1]
        # new_outputs = np.reshape(new_outputs, [batch_size, num_steps])
        feed = {model.inpt: new_inputs, model.keep_prob: 1.0, model.init_state_0: state1[0],
                model.init_state_1: state1[1],
                model.init_state_2: state2[0], model.init_state_3: state2[1]}
        state1[0], state1[1], state2[0], state2[1], probabilities = sess.run(
            [model.firstState, model.secondState, model.thirdState, model.fourthState, model.logits], feed)

        probabilities = np.exp(probabilities)
        probabilities = probabilities[0]
        magnitude = sum(probabilities)
        probabilities = probabilities / magnitude

        x += batch_size * num_steps
        max_index = np.random.choice(range(len(probabilities)), p=probabilities)
        most_likely_notes.append(max_index)
        previous_note = max_index

    return most_likely_notes


def notes_to_midi(notes):
    """
    Given a set of note tuples, creates a MIDI pattern with the notes in a single track.
    :param notes: A series of note tuples (pitch, velocity, duration)
    :return: A MIDI pattern
    """
    pattern = midi.Pattern()
    track = midi.Track()
    pattern.append(track)

    for i in range(len(notes)):
        pitch, velocity, tick = notes[i]
        track.append(midi.NoteOnEvent(pitch=pitch, velocity=velocity, tick=0))
        track.append(midi.NoteOffEvent(pitch=pitch, velocity=velocity, tick=tick))

    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)
    return pattern


def max_consecutive_length(seq):
    """
    Counts the length of the longest continuously rising (in increments of 1) subsequence in a given sequence
    :param seq: A sequence of integers
    :return: The length of the longest continuously rising subsequence
    """
    max_length = 1
    current_length = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1] + 1:
            current_length += 1
        else:
            max_length = max(max_length, current_length)
            current_length = 1

    max_length = max(max_length, current_length)
    return max_length


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='A generative music model.')
    parser.add_argument('--train', type=argparse.FileType('r'), nargs='+', help='MIDI files to use for training')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size for music model')
    parser.add_argument('--embedding_size', type=int, default=128, help='Embedding size for music model')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for training music model')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size for training music model')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for training music model')
    parser.add_argument('--num_steps', type=int, default=20,
                        help='Number of unrolling steps size for training music model')
    args = parser.parse_args()

    # load training data
    vocab, vocab_reverse = {}, []
    # list of tracks, each track will be a list of note IDs
    training_data = []
    training_files = args.train
    for f in training_files:
        pattern = midi.read_midifile(f)
        tracks = pattern[1:]
        # TODO currently we're only getting the first track
        track0 = tracks[0]
        notes, vocab, vocab_reverse = preprocess_track(track0, ids=(vocab, vocab_reverse))
        training_data.append(notes)

    model = MusicModel(hidden_size=args.hidden_size, embedding_size=args.embedding_size,
                       learning_rate=args.learning_rate, vocab_size=len(vocab))

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    train_model(sess, model, batch_size=args.batch_size, num_epochs=args.num_epochs, num_steps=args.num_steps,
                train_data=training_data)
    generated_notes = generate_music(sess, model, num_notes=100, note_context=training_data[0])
    print "Original Notes (first training track):"
    print training_data[0]
    print "Generated Notes:"
    print generated_notes

    gen_note_tuples = [vocab_reverse[token] for token in generated_notes]
    output_pattern = notes_to_midi(gen_note_tuples)
    print output_pattern[0][:6]
    midi.write_midifile("output.mid", output_pattern)


if __name__ == '__main__':
    main()
