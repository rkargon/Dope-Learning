"""
Functions for displaying statistics on sequences of notes
"""
import sys

import midi

from preprocess import get_notes_from_track


def note_stats(notes):
    """
    Creates histograms for the pitch, velocity, and duration distribution of a list of note tuples
    :param notes: A list of note tuples
    :return: A dictionary containing histograms for 'pitch', 'velocity', and 'duration'.
    Each histogram maps {value -> count}
    """
    pitches = [n[0] for n in notes]
    velocities = [n[1] for n in notes]
    durations = [n[2] for n in notes]

    stats = {'pitch': frequency_dict(pitches),
             'velocity': frequency_dict(velocities),
             'duration': frequency_dict(durations)}
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


def print_note_stats(*stats):
    """
    Displays a set of note stats.
    :param stats: A series of dictionaries with keys 'pitch', 'duration', and 'velocity', each mapping to a histogram.
    """
    for prop in ['pitch', 'duration', 'velocity']:
        print "Stats for %s:" % prop
        data = [s[prop] for s in stats]
        keys = sorted(set.union(set(), *[d.keys() for d in data]))
        for k in keys:
            print "%s:\t%s" % (k, "\t".join([str(d.get(k, 0)) for d in data]))


if __name__ == '__main__':
    fnames = sys.argv[1:]
    notes = [get_notes_from_track(midi.read_midifile(f)[1]) for f in fnames]
    note_stats = [note_stats(n) for n in notes]
    print_note_stats(*note_stats)
