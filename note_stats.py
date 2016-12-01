"""
Functions for displaying statistics on sequences of notes
"""
import sys

import midi

from preprocess import get_events_from_track, event_tuples_to_notes

# TODO perhaps encapsulate this in a NoteStats class
NOTE_STATS = ['pitch', 'velocity', 'duration', 'start_time']


def note_stats(notes):
    """
    Creates histograms for the type, pitch, and tick distribution of a list of note tuples
    :param notes: A list of note tuples
    :return: A dictionary containing histograms for 'type', 'pitch', and 'tick'.
    Each histogram maps {value -> count}
    """

    pitches, velocities, durations, start_times = zip(*notes)
    stats = {
        'pitch': frequency_dict(pitches),
        'velocity': frequency_dict(velocities),
        'duration': frequency_dict(durations),
        'start_time': frequency_dict(start_times)}
    return stats


def frequency_dict(l):
    """
    Given a list of elements, makes a frequency dictionary mapping values to their count in the given list.
    :param l: A list of elements
    :return: A dictionary mapping element -> count
    """
    d = {}
    for e in l:
        d[e] = d.get(e, 0) + 1
    return d


def print_note_stats(stats, stats_to_print=NOTE_STATS):
    """
    Displays a set of note stats.
    :param stats: A list of dictionaries with keys 'pitch', 'duration', and 'velocity', each mapping to a histogram.
    :param stats_to_print: A list of keywords to print stats for. By default, it uses NOTE_STATS.
    """
    for prop in stats_to_print:
        print "Stats for %s:" % prop
        data = [s[prop] for s in stats]
        keys = sorted(set.union(set(), *[d.keys() for d in data]))
        for k in keys:
            print "%s:\t%s" % (k, "\t".join([str(d.get(k, 0)) for d in data]))


def main():
    # TODO argparse for this
    if len(sys.argv) == 1:
        print "note_stats.py <midi files...>\tThis script prints out stats for a series of MIDI files. The output is " \
              "a table with a column for each input MIDI file."
        exit()
    fnames = sys.argv[1:]
    # get MIDI events from first track in each file (as a separate list for each file)
    # TODO our generated data puts music on 0th track, not 1st.
    events = [get_events_from_track(midi.read_midifile(f)[1]) for f in fnames]
    notes = [event_tuples_to_notes(es) for es in events]
    stats = [note_stats(n) for n in notes]
    print_note_stats(stats, stats_to_print=['pitch', 'duration'])


if __name__ == '__main__':
    main()
