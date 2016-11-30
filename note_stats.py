"""
Functions for displaying statistics on sequences of notes
"""
import sys

import midi

from preprocess import get_notes_from_track


# TODO parse actual notes instead of events to get info about duration
def note_stats(notes):
    """
    Creates histograms for the type, pitch, and tick distribution of a list of event tuples
    :param notes: A list of event tuples
    :return: A dictionary containing histograms for 'type', 'pitch', and 'tick'.
    Each histogram maps {value -> count}
    """
    types = [n[0] for n in notes]
    pitches = [n[1] for n in notes]
    ticks = [n[2] for n in notes]

    stats = {'type': frequency_dict(types),
             'pitch': frequency_dict(pitches),
             'tick': frequency_dict(ticks)}
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
    for prop in ['type', 'pitch', 'tick']:
        print "Stats for %s:" % prop
        data = [s[prop] for s in stats]
        keys = sorted(set.union(set(), *[d.keys() for d in data]))
        for k in keys:
            print "%s:\t%s" % (k, "\t".join([str(d.get(k, 0)) for d in data]))


def main():
    """
    Function for testing
    """
    fnames = sys.argv[1:]
    notes = [get_notes_from_track(midi.read_midifile(f)[1]) for f in fnames]
    stats = [note_stats(n) for n in notes]
    print_note_stats(*stats)


if __name__ == '__main__':
    main()
