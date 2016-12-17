"""
Functions for converting between MIDI data and formats used by the music model, such as note tuples or note IDs.
"""

import midi
import numpy as np


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

# ------------------------------------------------------------------------------
# The following functions are from: https://github.com/dshieble/Musical_Matrices
# Author: Dan Shiebler (danshiebler@gmail.com)

lowerBound = 24
upperBound = 102
span = upperBound-lowerBound

def midiToNoteStateMatrix(midifile, squash=True, span=span):
    pattern = midi.read_midifile(midifile)

    timeleft = [track[0].tick for track in pattern]

    posns = [0 for track in pattern]

    statematrix = []
    time = 0

    state = [[0,0] for x in range(span)]
    statematrix.append(state)
    condition = True
    while condition:
        if time % (pattern.resolution / 4) == (pattern.resolution / 8):
            # Crossed a note boundary. Create a new state, defaulting to holding notes
            oldstate = state
            state = [[oldstate[x][0],0] for x in range(span)]
            statematrix.append(state)
        for i in range(len(timeleft)): #For each track
            if not condition:
                break
            while timeleft[i] == 0:
                track = pattern[i]
                pos = posns[i]

                evt = track[pos]
                if isinstance(evt, midi.NoteEvent):
                    if (evt.pitch < lowerBound) or (evt.pitch >= upperBound):
                        pass
                        # print "Note {} at time {} out of bounds (ignoring)".format(evt.pitch, time)
                    else:
                        if isinstance(evt, midi.NoteOffEvent) or evt.velocity == 0:
                            state[evt.pitch-lowerBound] = [0, 0]
                        else:
                            state[evt.pitch-lowerBound] = [1, 1]
                elif isinstance(evt, midi.TimeSignatureEvent):
                    if evt.numerator not in (2, 4):
                        # We don't want to worry about non-4 time signatures. Bail early!
                        # print "Found time signature event {}. Bailing!".format(evt)
                        out =  statematrix
                        condition = False
                        break
                try:
                    timeleft[i] = track[pos + 1].tick
                    posns[i] += 1
                except IndexError:
                    timeleft[i] = None

            if timeleft[i] is not None:
                timeleft[i] -= 1

        if all(t is None for t in timeleft):
            break

        time += 1

    S = np.array(statematrix)
    statematrix = np.hstack((S[:, :, 0], S[:, :, 1]))
    statematrix = np.asarray(statematrix).tolist()
    return statematrix

def noteStateMatrixToMidi(statematrix, name="example", span=span):
    statematrix = np.array(statematrix)
    if not len(statematrix.shape) == 3:
        statematrix = np.dstack((statematrix[:, :span], statematrix[:, span:]))
    statematrix = np.asarray(statematrix)
    pattern = midi.Pattern()
    track = midi.Track()
    pattern.append(track)
    
    span = upperBound-lowerBound
    tickscale = 55
    
    lastcmdtime = 0
    prevstate = [[0,0] for x in range(span)]
    for time, state in enumerate(statematrix + [prevstate[:]]):  
        offNotes = []
        onNotes = []
        for i in range(span):
            n = state[i]
            p = prevstate[i]
            if p[0] == 1:
                if n[0] == 0:
                    offNotes.append(i)
                elif n[1] == 1:
                    offNotes.append(i)
                    onNotes.append(i)
            elif n[0] == 1:
                onNotes.append(i)
        for note in offNotes:
            track.append(midi.NoteOffEvent(tick=(time-lastcmdtime)*tickscale, pitch=note+lowerBound))
            lastcmdtime = time
        for note in onNotes:
            track.append(midi.NoteOnEvent(tick=(time-lastcmdtime)*tickscale, velocity=40, pitch=note+lowerBound))
            lastcmdtime = time
            
        prevstate = state
    
    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)

    midi.write_midifile("{}.mid".format(name), pattern)
