"""
Functions for converting between MIDI data and formats used by the music model, such as note tuples or note IDs.
"""

import midi

# TODO store notes in a class? This could unify the definition and make things clearer

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


def notes_to_midi(notes, resolution=None):
    """
    Given a set of note tuples, creates a MIDI pattern with the notes in a single track.
    :param resolution: The resolution for the created MIDI pattern. If None, uses pyMIDI's default value.
    :param notes: A series of note tuples (pitch, velocity, duration)
    :return: A MIDI pattern
    """
    if resolution is not None:
        pattern = midi.Pattern(resolution=resolution)
    else:
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
