'''
input: name of midi track
output: proportion of sequential pitches that have the same value
'''

import midi
import sys
from preprocess import preprocess_track

tracks = midi.read_midifile(sys.argv[1])

pitch_score = 0
num_pitches = 0
for t in tracks:
    for i in range(0, len(t) - 3, 2):
        event1 = t[i]
        event2 = t[i + 2]
        if issubclass(type(event1), midi.NoteEvent) and issubclass(type(event1), midi.NoteEvent):
            num_pitches += 1
            pitch1 = t[i].data[0]
            pitch2 = t[i + 2].data[0]
            if pitch1 == pitch2:
                pitch_score += 1
print 1.0 * pitch_score / num_pitches
