import tensorflow as tf
import sys # for now?
import codecs
import pickle
import numpy as np

from nlptools import unk, STOP, UNK

# hyper-parameters:
batch_size = 50
hidden_size = 256
embed_size = 50
seq_length = 20
dropout = 0.5

""" This can generate text as well as evaluate perplexity """
class LSTMLangmod:

  """ train_file: path to text corpus, or path to saved model.
      saved: if saved is False (default), train_file is a path to corpus from which to train
              otherwise try to load the graph from that path
  """
  def __init__(self, train_file, saved=True):
    self.sess = tf.Session()
    self._training = False
    if saved:
      with open("%s.dict"%train_file, "rb") as f:
        self._vocab = pickle.load(f)
        
      self.vocab_size = len(self._vocab.keys())

      saver = tf.train.import_meta_graph("%s.meta"%train_file)
      saver.restore(self._sess, train_file) # restore the session

      self._inpt = tf.get_collection('input')[0]
      self._output = tf.get_collection('targets')[0]
      self._logits = tf.get_collection('logits')[0]
      self._cross_entropy = tf.get_collection('cross_entropy')[0]
    else:
      self._training = True
      self._train = train_file

      text_corpus = self._processCorpus()
      self._vocab = self._makeWordIDs(text_corpus) # map word to int id
      self._corpus = [self._vocab[w] for w in text_corpus]
      self.vocab_size = len(self._vocab.keys()) + 1 # include invalid word

      self._inv_map = {v: k for k, v in self._vocab.items()} # inverse map int id->string
      print(self.vocab_size)

      # inputs and outputs:
      self._inpt = tf.placeholder(tf.int32, [None])
      self._targets = tf.placeholder(tf.int32, [None])
      self._keep_prob = tf.placeholder(tf.float32)

      # word embeddings:
      E = tf.Variable(tf.truncated_normal([self.vocab_size, embed_size], stddev=0.1))
      Elookup = tf.nn.embedding_lookup(E, tf.reshape(self._inpt, [batch_size, seq_length]))

      # start describing our RNN:
      self.lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, state_is_tuple=True)
      state_size = self.lstm_cell.state_size
      self._istate0 = tf.placeholder(tf.float32, [None,state_size[0]], name="istate0")
      self._istate1 = tf.placeholder(tf.float32, [None,state_size[1]], name="istate1")
      init_state = tf.nn.rnn_cell.LSTMStateTuple(self._istate0, self._istate1)

      # simulate time steps:
      drop = tf.nn.dropout(Elookup, keep_prob=self._keep_prob)
      output, self._state = tf.nn.dynamic_rnn(self.lstm_cell, drop, dtype=tf.float32, initial_state=init_state)

      #self._lstm_out = tf.placeholder(tf.float32, [batch_size, seq_length, hidden_size])
      output = tf.reshape(output, [-1, hidden_size])

      # softmax layer:
      sm_weights = tf.Variable(tf.truncated_normal([hidden_size, self.vocab_size], stddev=0.1))
      sm_biases = tf.Variable(tf.constant(0.1, shape=[self.vocab_size]))
      self._logits = tf.matmul(output, sm_weights) + sm_biases # batch_size x vocab_size
      w = tf.ones(tf.shape(batch_size*seq_length))
      self._loss = tf.nn.seq2seq.sequence_loss_by_example([self._logits], [self._targets], [w])

      # define the training step:
      self._train_step = tf.train.AdamOptimizer(1e-4).minimize(self._loss)

      # setup the tf session:
      self.sess.run(tf.initialize_all_variables())

    # for inference and analysis:
    self._losshape = tf.shape(self._loss)
    self._perplexity = tf.exp(tf.reduce_sum(self._loss)/(batch_size*seq_length))
    self._probs = tf.nn.softmax(self._logits) # convert to probabilities

    self._inv_map = {v: k for k, v in self._vocab.items()} # inverse map int id->string

    self.istate = self.sess.run(self.lstm_cell.zero_state(batch_size, tf.float32))  # initial state

  def train(self):
    if self._training:
      n = 0
      window_size = batch_size*seq_length
      iter_range = range(0,len(self._corpus)-window_size-1,window_size)
      total = len(iter_range)

      for i in iter_range:
        words = self._corpus[i:i + window_size]
        nextwords = self._corpus[i + 1:i + window_size + 1]

        self.sess.run(self._train_step,feed_dict={self._inpt: words,
                                                  self._targets: nextwords,
                                                  self._keep_prob: dropout,
                                                  self._istate0: self.istate.c,
                                                  self._istate1: self.istate.h})
        self.istate = self.sess.run(self._state,
                               feed_dict={self._inpt: words,
                                          self._keep_prob: dropout,
                                          self._istate0: self.istate.c,
                                          self._istate1: self.istate.h})
        if not n%100:
          #p = self.sess.run(self._perplexity, feed_dict={self._lstm_out: output, self._targets: nextwords})
          p = self.sess.run(self._perplexity,feed_dict={self._inpt: words,
                                                        self._targets: nextwords,
                                                        self._keep_prob: dropout,
                                                        self._istate0: self.istate.c,
                                                        self._istate1: self.istate.h})
          print("Batch #%d of %d (%.2f%%): %.4f"%(n,total,100*n/total,p))

        n+=1

  """ Save a model to a path, returns the path to which it was saved """
  def saveModel(self, path):
    tf.add_to_collection('logits', self._logits)
    tf.add_to_collection('input', self._inpt)
    tf.add_to_collection('targets', self._targets)
    tf.add_to_collection('cross_entropy', self._cross_entropy)
    saver = tf.train.Saver()

    with open("%s.dict"%path, "wb") as f:
      pickle.dump(self._vocab, f)
    #with open("%s.unk"%path, "wb") as f:
    #  pickle.dump(self.unker, f)

    p = saver.save(self.sess, path)
    return p # the filename under which the model was saved

  """ Generate sentence of length n """
  def generate(self, n):
    stopcode = self._vocab[STOP]
    # sentence = [ np.random.randint(vocab_size) ]
    sentence = [ stopcode ]
    while len(sentence) < n:
      words = sentence + [0]*(seq_length - len(sentence)) # pad with zeros
      print(words)
      dist = np.array(self._probs.eval(feed_dict={self._inpt: [words]}, session=self.sess)[0])
      dist /= dist.sum()
      nword = np.random.choice(len(dist),p=dist)

      if nword == stopcode:
        break
      elif nword == 0:
        continue
      sentence.append(nword)

    s = sentence[1:] # exclude stop symbol
    return "".join([self._inv_map.get(w, "")+" " for w in s]).strip()

  """ Evaluate perplexity of input
        text: list of words as strings
  """
  def evaluate(self, text):
    u = self._vocab[UNK] # unk as int
    text = [self._vocab.get(w, u) for w in text] # map to integers
    n = 0
    window_size = batch_size * seq_length
    iter_range = range(0, len(text) - window_size - 1, window_size)
    total = len(iter_range)

    perplexes = np.zeros(len(iter_range))
    for i in iter_range:
      words = text[i:i + window_size]
      nextwords = text[i + 1:i + window_size + 1]

      perplexes[n] = self.sess.run(self._perplexity, feed_dict={self._inpt: words,
                                                                self._targets: nextwords,
                                                                self._keep_prob: 1.0,
                                                                self._istate0: self.istate.c,
                                                                self._istate1: self.istate.h})
      #if not n%100:
      print("Batch #%d of %d (%.2f%%): %.4f" % (n, total, 100 * n / total,perplexes[n]))

      n += 1

    #print(perplexes)
    return np.sum(perplexes)/np.size(perplexes)

  """ Pad a list with zeros """
  def _padList(self, l):
    return l + [0]*(batch_size*seq_length - len(l))

  """ Make the mapping of a word to unique integer id """
  def _makeWordIDs(self, text):
    wordIDs = {}
    index = 1
    for word in text:
      if word not in wordIDs:
        wordIDs[word] = index
        index += 1

    return wordIDs # maps word -> int

  """ Read in a tokenised text file, unk it and return list of words """
  def _processCorpus(self):
    with codecs.open(self._train, "r",encoding='utf-8', errors='ignore') as f:
      # text = [("%s %s %s"%(STOP,line,STOP)).split() for line in f]
      text = [ STOP ]
      for line in f:
        text.extend((line+" "+STOP).split())

    counts = {} # map word->count
    for w in text:
      counts[w] = counts.get(w, 0) + 1 # increment count of this word

    unker = unk.BasicUnker(text, counts) # unk anything with count <= 1
    corpus = unker.getUnkedCorpus() # does what it says

    return corpus

