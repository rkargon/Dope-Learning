import tensorflow as tf
import midi
import numpy as np
import collections
import math
import operator

midiFile = midi.read_midifile('Dope-Learning-master/midik/1080-01.mid')
tracks = midiFile[1:len(midiFile)]
track_lengths = list()
allNotes = list()
for j in range(len(tracks)):
	track0 = tracks[j]
	track_lengths.append(len(tracks[j]))
	track0 = track0[1:len(track0)-1:2]
	for i in range(len(track0)):
		noteEvent = track0[i]
		track0[i] = (noteEvent.pitch, noteEvent.tick)
		allNotes.append(track0[i])

notes = list()

counter = collections.Counter(allNotes)
count_pairs = sorted(counter.items(), key = lambda x: (-x[1], x[0]))
notes, _ = list(zip(*count_pairs))
notes_to_id = dict(zip(notes, range(len(notes))))
vocab_size = len(notes_to_id)

train_notes_ids = [notes_to_id[note] for note in allNotes]

batch_size = 5
num_steps = 5
hidden_size = 128
embed_sz = 128
learning_rate = 0.0001

#inpt = tf.placeholder(tf.int32, [batch_size, num_steps])
#output = tf.placeholder(tf.int32, [batch_size, num_steps])
inpt = tf.placeholder(tf.int32, [None, None])
output = tf.placeholder(tf.int32, [None, None])
keep_prob = tf.placeholder(tf.float32)

E = tf.Variable(tf.random_uniform([vocab_size, embed_sz], -1.0, 1.0))
embd = tf.nn.embedding_lookup(E, inpt)

lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, state_is_tuple = True)
num_layers = 2
cell = tf.nn.rnn_cell.MultiRNNCell([lstm]*num_layers, state_is_tuple=True)
init_state_0 = tf.placeholder(tf.float32, [None, hidden_size])
init_state_1 = tf.placeholder(tf.float32, [None, hidden_size])
init_state_2 = tf.placeholder(tf.float32, [None, hidden_size])
init_state_3 = tf.placeholder(tf.float32, [None, hidden_size])
firstTuple = tf.nn.rnn_cell.LSTMStateTuple(init_state_0, init_state_1)
secondTuple = tf.nn.rnn_cell.LSTMStateTuple(init_state_2, init_state_3)
init_state = (firstTuple, secondTuple)
W1 = tf.Variable(tf.truncated_normal([hidden_size, vocab_size], stddev = 0.1))
B1 = tf.Variable(tf.random_uniform([vocab_size], -1.0, 1.0))

new_embeddings = tf.nn.dropout(embd, keep_prob)
outputs, state = tf.nn.dynamic_rnn(cell, new_embeddings, initial_state = init_state)
firstState = state[0][0]
secondState = state[0][1]
thirdState = state[1][0]
fourthState = state[1][1]
dimensions = tf.slice(tf.shape(inpt), [0], [1])
firstDim = tf.reshape(dimensions, [])
dimensions = tf.slice(tf.shape(inpt), [1], [1])
secondDim = tf.reshape(dimensions, [])
reshapedOutput = tf.reshape(outputs, [firstDim*secondDim, hidden_size])
logits = tf.matmul(reshapedOutput, W1)+B1

loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [tf.reshape(output, [-1])], [tf.ones([firstDim*secondDim])])
loss = tf.reduce_sum(loss)
loss = loss/tf.to_float(firstDim)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

num_epochs = 100
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

state1 = sess.run(lstm.zero_state(batch_size, tf.float32))
state2 = sess.run(lstm.zero_state(batch_size, tf.float32))
for i in range(num_epochs):
	print ('\n')
	print ("Epoch %d!!!" % (i+1))	
	total_error = 0.0
	x = 0
	count = 0
	state1 = sess.run(lstm.zero_state(batch_size, tf.float32))
	tmp = list()
	tmp.append(state1[0])
	tmp.append(state1[1])
	state1 = tmp
	state2 = sess.run(lstm.zero_state(batch_size, tf.float32))
	tmp2 = list()
	tmp2.append(state2[0])
	tmp2.append(state2[1])
	state2 = tmp2
	while((x+batch_size*num_steps) < len(train_notes_ids)):
		count += num_steps
		inputs = train_notes_ids[x:x+batch_size*num_steps]
		inputs = np.reshape(inputs, [batch_size, num_steps])
		outputs = train_notes_ids[x+1:x+batch_size*num_steps+1]
		outputs = np.reshape(outputs, [batch_size, num_steps])
		x += batch_size*num_steps
		feed = {inpt: inputs, output: outputs, keep_prob: 0.5, init_state_0: state1[0], init_state_1: state1[1], init_state_2: state2[0], init_state_3: state2[1]}
		err, state1[0], state1[1], state2[0], state2[1], probabilities, _ = sess.run([loss, firstState, secondState, thirdState, fourthState, logits, train_step], feed) 
		total_error += err
		print math.exp(total_error/count)
		
		probabilities = np.exp(probabilities)
		magnitude = np.linalg.norm(probabilities)
		probabilities = probabilities/magnitude
		#print np.shape(probabilities) 

inputs = train_notes_ids[0:len(train_notes_ids)-1]
outputs = train_notes_ids[1:len(train_notes_ids)]

batch_size = 1 
num_steps = 1
x = 0
count = 0
total_error = 0.0
state1 = sess.run(lstm.zero_state(batch_size, tf.float32))
tmp = list()
tmp.append(state1[0])
tmp.append(state1[1])
state1 = tmp
state2 = sess.run(lstm.zero_state(batch_size, tf.float32))
tmp = list()
tmp.append(state2[0])
tmp.append(state2[1])
state2 = tmp
previousNote = -1
mostLikelyNotes = list()
while((x+batch_size*num_steps) < len(train_notes_ids)-1):
	count += num_steps
	if(previousNote == -1):
		new_inputs= inputs[x:x+batch_size*num_steps]
		new_inputs = np.reshape(new_inputs, [batch_size, num_steps])
	else:
		new_inputs = [max_index]
		new_inputs = np.reshape(new_inputs, [batch_size, num_steps])
	new_outputs = outputs[x+1:x+batch_size*num_steps+1]
	new_outputs = np.reshape(new_outputs, [batch_size, num_steps])
	feed = {inpt: new_inputs, output: new_outputs, keep_prob: 1.0, init_state_0: state1[0], init_state_1: state1[1], init_state_2: state2[0], init_state_3: state2[1]}
	err, state1[0], state1[1], state2[0], state2[1], probabilities= sess.run([loss, firstState, secondState, thirdState, fourthState, logits], feed)
	total_error += err
	print math.exp(total_error/count)
	probabilities = np.exp(probabilities)
	probabilities = probabilities[0]
	magnitude = sum(probabilities)
	probabilities = probabilities/magnitude
	print probabilities
	x += batch_size*num_steps
	max_index = np.random.choice(range(len(probabilities)), p=probabilities)
	mostLikelyNotes.append(max_index) 
	previousNote = max_index

print mostLikelyNotes
print train_notes_ids

id_to_note = {v:k for k,v in notes_to_id.iteritems()}
tuples = list()
for i in range(len(mostLikelyNotes)):
	tuples.append(id_to_note[mostLikelyNotes[i]])
print tuples

pattern = midi.Pattern()	
track = midi.Track()
pattern.append(track)

trackWeAreOn = 0
count = 0
for i in range(len(tuples)):
	count += 1
	tuplei = tuples[i]
	on = midi.NoteOnEvent(tick = 0, velocity = 100, pitch = tuplei[0])
	track.append(on)
	off = midi.NoteOffEvent(tick = tuplei[1], pitch = tuplei[0])	
	track.append(off)
	if count == track_lengths[trackWeAreOn]-1:
		track = midi.Track()
		pattern.append(track)
	
eot = midi.EndOfTrackEvent(tick=1)
track.append(eot)
midi.write_midifile("output.mid", pattern)
