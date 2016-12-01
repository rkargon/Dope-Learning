"""
Functions for converting between MIDI data and formats used by the music model, such as note tuples or note IDs.
"""

import midi


# TODO store notes in a class? This could unify the definition and make things clearer

def midi_event_to_tuple(e):
    """
    Takes a MIDI note event and converts it to tuple form. This is used primarily to get immutable event objects that
    can be hashed effectively.
    :param e: A given midi event
    :return: (event_type, pitch, tick), or None if the event is not a midi.NoteEvent.
    """
    if not issubclass(type(e), midi.NoteEvent):
        return None
    return type(e), e.pitch, e.tick, e.velocity


def tuple_to_midi_event(et):
    """
    Converts an event tuple into a MIDI event
    :param et: The given event tuple
    :return: A MIDI NoteEvent
    """
    event_type, pitch, tick, velocity = et
    return event_type(pitch=pitch, tick=tick, velocity=velocity)


def event_tuples_to_notes(events):
    """
    Converts a list of MIDI event tuples into a series of notes. Each note has a pitch, MIDI 'velocity' (roughly
    analogous to volume), duration, and start time.
    :param events: A list of MIDI NoteOn or NoteOff event tuples
    :return: A list of note tuples, sorted by start time.
    """
    notes_map = [None] * 128
    notes = []
    current_tick = 0
    for e in events:
        e_type, e_pitch, e_tick, e_velocity = e
        current_tick += e_tick
        if e_type is midi.NoteOnEvent:
            notes_map[e_pitch] = (e_pitch, e_velocity, current_tick)
        else:
            # check if NoteOff event is triggered for note that is already off
            # This only happens in our generated MIDI files, which might have imperfect structure.
            if notes_map[e_pitch] is None:
                continue
            _, _, note_start_time = notes_map[e_pitch]
            note = (e_pitch, e_velocity, current_tick - note_start_time, note_start_time)
            notes.append(note)
            notes_map[e_pitch] = None
    notes.sort(key=lambda n: n[3])
    return notes


def get_events_from_track(track):
    """
    Extracts NoteOn and NoteOff events from a MIDI track
    :param track: A MIDI track
    :return: A list of NoteOn and NoteOff events
    """
    return filter(None, [midi_event_to_tuple(e) for e in track])


def preprocess_track(track, ids=None):
    """
    Takes a MIDI track, maps them to a series of note tuples, and returns a list of their IDs (along with the mapping
    of note -> ID)
    :param track: A MIDI track
    :param ids: An existing note->ID mapping can be specified with a tuple (vocab, vocab_reverse).
    :return: The track as a sequence of note IDs, a dictionary mapping notes to IDs, and a list where each note's ID
    is its index.
    """
    return build_vocabulary(get_events_from_track(track), ids=ids)


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


def events_to_midi(events, resolution=None):
    """
    Given a set of MIDI event tuples, creates a MIDI pattern with the events in a single track.
    :param resolution: The resolution for the created MIDI pattern. If None, uses pyMIDI's default value.
    :param events: A series of NoteOn or NoteOff events
    :return: A MIDI pattern
    """
    if resolution is not None:
        pattern = midi.Pattern(resolution=resolution)
    else:
        pattern = midi.Pattern()
    track = midi.Track(events=[tuple_to_midi_event(e) for e in events])
    eot = midi.EndOfTrackEvent(tick=1)
    pattern.append(track)
    track.append(eot)
    return pattern
