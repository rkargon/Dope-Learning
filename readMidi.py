import tensorflow as tf
import midi
import numpy as np

midiFile = midi.read_midifile('Dope-Learning-master/midik/1080-01.mid')
#print midiFile
tracks = midiFile[1:len(midiFile)]
#First track 
print len(tracks[0])

#NoteEvent
print tracks[0][0]

event = tracks[0][0]

#get the note, can find out which one it is from table
note = event.pitch

print note




