#!/usr/bin/env python2.7
import argparse
import logging

import midi
import numpy as np
import tensorflow as tf
import time

from note_stats import note_stats, print_note_stats
from preprocess import preprocess_track, events_to_midi, event_tuples_to_notes, generate_rising_scale, build_vocabulary

# get logger for current script (even across different modules)
logger = logging.getLogger(__name__)


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
        first_tuple = tf.nn.rnn_cell.LSTMStateTuple(self.init_state_0, self.init_state_1)
        second_tuple = tf.nn.rnn_cell.LSTMStateTuple(self.init_state_2, self.init_state_3)
        self.init_state = (first_tuple, second_tuple)

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
        reshaped_output = tf.reshape(outputs, [first_dim * second_dim, hidden_size])
        self.logits = tf.matmul(reshaped_output, W1) + B1

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


# def run_model(sess, model, inputs):
#     pass


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
        logger.info("Training epoch %d of %d..." % (i + 1, num_epochs))
        start_time = time.time()
        total_error = 0.0
        for track in train_data:
            x = 0
            state1 = list(sess.run(model.lstm.zero_state(batch_size, tf.float32)))
            state2 = list(sess.run(model.lstm.zero_state(batch_size, tf.float32)))
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

        logger.info("Total error: %f" % total_error)
        elapsed_time = time.time() - start_time
        logger.info("Elapsed time: %f seconds..." % elapsed_time)


# TODO this shares a lot of code with training, we might be able to abstract some of this out
# TODO only first note of note_context is used right now
def generate_music(sess, model, num_notes, note_context):
    """
    Uses a trained model to generate notes of music one by one.
    :param sess: The tensorflow session with which to run the model
    :param model: The given music model
    :param num_notes: The number of notes to generate
    :param note_context: A context of notes to feed into the model before generating new ones
    :return: A series of generated note IDs
    """
    batch_size = 1
    num_steps = 1
    x = 0
    count = 0
    state1 = list(sess.run(model.lstm.zero_state(batch_size, tf.float32)))
    state2 = list(sess.run(model.lstm.zero_state(batch_size, tf.float32)))
    previous_note = -1
    max_index = None
    most_likely_notes = list()
    while (x + batch_size * num_steps) < num_notes - 1:
        count += num_steps
        if previous_note == -1:
            new_inputs = note_context[x:x + batch_size * num_steps]
            new_inputs = np.reshape(new_inputs, [batch_size, num_steps])
        else:
            new_inputs = [max_index]
            new_inputs = np.reshape(new_inputs, [batch_size, num_steps])
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


def save_model(sess, path, saver=None):
    """
    Saves a tensorflow session to the given path.
    NOTE: This currently saves *all* variables in the session, unless one passes in a custom Saver object.
    :param sess: The tensorflow session to save from
    :param path: The path to store the saved data
    :param saver: A custom saver object to use. This can be used to only save certain variables. If None,
    creates a saver object that saves all variables.
    :return: The saver object used.
    """
    if saver is None:
        saver = tf.train.Saver(tf.all_variables())
    saver.save(sess, path)
    return saver


def restore_model(sess, path, saver=None):
    """
    Loads a tensorflow session from the given path.
    NOTE: This currently loads *all* variables in the saved file, unless one passes in a custom Saver object.
    :param sess: The tensorflow checkpoint to load from
    :param path: The path to the saved data
    :param saver: A custom saver object to use. This can be used to only load certain variables. If None,
    creates a saver object that loads all variables.
    :return: The saver object used.
    """
    if saver is None:
        saver = tf.train.Saver(tf.all_variables())
    saver.restore(sess, path)
    return saver


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='A generative music model.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train', type=argparse.FileType('r'), nargs='+', help='MIDI files to use for training',
                        required=True)
    parser.add_argument('--test_song', type=argparse.FileType('r'), nargs='+', help='MIDI song to use for testing')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size for music model')
    parser.add_argument('--embedding_size', type=int, default=128, help='Embedding size for music model')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for training music model')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size for training music model')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for training music model')
    parser.add_argument('--num_steps', type=int, default=25,
                        help='Number of unrolling steps size for training music model')
    parser.add_argument('-o', '--output', type=argparse.FileType('w'), default='output.mid',
                        help='file to write music output to.')
    parser.add_argument('--model_load_path', type=str, default=None,
                        help='file to load a saved model from.')
    parser.add_argument('--model_save_path', type=str, default='model.ckpt',
                        help='file to save a trained model to.')
    parser.add_argument('-l', '--log', action='store_true', default=True, help='Print out progress and other info')
    args = parser.parse_args()

    if args.log:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    # load training data
    vocab, vocab_reverse = {}, []
    # list of tracks, each track will be a list of note IDs
    training_data = []
    training_files = args.train
    track_resolution = None
    logger.info("Loading training data... (%d files)" % len(training_files))
    for f in training_files:
        pattern = midi.read_midifile(f)
        if track_resolution is None:
            track_resolution = pattern.resolution
        tracks = pattern[1:]
        for track in tracks:
            notes, vocab, vocab_reverse = preprocess_track(track, ids=(vocab, vocab_reverse))
            training_data.append(notes)
    logger.info("Read %d tracks, with %d total notes and %d unique notes." % (
        len(training_data), sum([len(t) for t in training_data]), len(vocab)))

    # notes, vocab, vocab_reverse = build_vocabulary(generate_rising_scale())
    # training_data = [notes]

    logger.info("Initializing model...")
    model = MusicModel(hidden_size=args.hidden_size, embedding_size=args.embedding_size,
                       learning_rate=args.learning_rate, vocab_size=len(vocab))

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    # train model
    if args.model_load_path is not None:
        logger.info("Loading model from path %s ..." % args.model_load_path)
        restore_model(sess, args.model_load_path)

    logger.info("Training model...")
    train_model(sess, model, batch_size=args.batch_size, num_epochs=args.num_epochs, num_steps=args.num_steps,
                train_data=training_data)
    logger.info("Saving trained model to %s ..." % args.model_save_path)
    save_model(sess, args.model_save_path)

    # generate notes
    if args.test_song is not None:
        pattern = midi.read_midifile(args.test_song)
        tracks = pattern[1:]
        notes, vocab, vocab_reverse = preprocess_track(tracks[0], ids=(vocab, vocab_reverse))
        test_track = notes
    else:
        test_track = training_data[0]
    generated_notes = generate_music(sess, model, num_notes=len(training_data[0]), note_context=test_track)
    logger.info("Original Notes (first training track):")
    logger.info(training_data[0])
    logger.info("Generated Notes:")
    logger.info(generated_notes)

    # write generated music to MIDI file
    gen_event_tuples = [vocab_reverse[token] for token in generated_notes]
    output_pattern = events_to_midi(gen_event_tuples, resolution=track_resolution)
    midi.write_midifile(args.output, output_pattern)

    # print stats for data
    logger.info("Original, Generated Stats:")
    original_event_tuples = [vocab_reverse[t] for t in training_data[0]]
    original_notes = event_tuples_to_notes(original_event_tuples)
    generated_notes = event_tuples_to_notes(gen_event_tuples)
    stats = [note_stats(original_notes), note_stats(generated_notes)]
    print_note_stats(stats, stats_to_print=['pitch', 'duration'])

    # write reconstruction of original MIDI file to disk
    logger.info(track_resolution)
    midi_pattern = events_to_midi([vocab_reverse[t] for t in training_data[0]], resolution=track_resolution)
    midi.write_midifile("reconstructed.mid", midi_pattern)


if __name__ == '__main__':
    main()
