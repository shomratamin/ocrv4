
from __future__ import division
import math
from six.moves import xrange
import tensorflow as tf
import numpy as np
import os
from skimage.transform import resize as imresize
import cv2
import time

from tqdm import tqdm
import collections
from tensorflow.contrib.rnn import RNNCell, LSTMStateTuple
from sklearn.model_selection import train_test_split
import csv
import random

os.environ['CUDA_VISIBLE_DEVICES'] = '0'




def read_dataset2(filename):
    output = []
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
        lines = text.split('\n')
        for _data in lines:
            data = _data.split(':',1)
            if len(data) >= 2:
                image_base_name = data[0]
                _line_text = data[1]
                output.append([image_base_name,_line_text])

    return output

def read_dataset(filename):
    output = []
    with open(filename, 'r', encoding='utf-8') as f:
        lines = csv.reader(f, delimiter=',', quotechar='"')

        for line in lines:
            output.append(line)
    return output


directory = 'ocrv4_dataset'
images = []
labels = []
data = read_dataset(f'{directory}/annot_cleaned.csv')
for line in data:
    image_path = f'{directory}/{line[0]}'
    text = line[1]
    images.append(image_path)
    labels.append(text)

print(len(images))
print(len(labels))



# charset = list(set(''.join(labels)))
charset = ' "#&\'()*,-./0123456789:;ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]_abcdefghijklmnopqrstuvwxyz}{ʼ'
num_classes = len(charset) + 2
encode_maps = {}
decode_maps = {}
for i, char in enumerate(charset, 3):
    encode_maps[char] = i
    decode_maps[i] = char
    
SPACE_INDEX = 0
SPACE_TOKEN = '<PAD>'
encode_maps[SPACE_TOKEN] = SPACE_INDEX
decode_maps[SPACE_INDEX] = SPACE_TOKEN

GO_INDEX = 1
GO_TOKEN = '<GO>'
encode_maps[GO_TOKEN] = GO_INDEX
decode_maps[GO_INDEX] = GO_TOKEN

EOS_INDEX = 2
EOS_TOKEN = '<EOS>'
encode_maps[EOS_TOKEN] = EOS_INDEX
decode_maps[EOS_INDEX] = EOS_TOKEN


GO = 1
PAD = 0
EOS = 2


image_height = 32
image_width = 480
image_channel = 1
max_stepsize = 128
num_hidden = 256
epoch = 28
batch_size = 128
initial_learning_rate = 1e-3




X, Y = [], []
for i in tqdm(range(len(images))):
    img = images[i]
    __image = cv2.imread(img)
    __image = cv2.cvtColor(__image,cv2.COLOR_BGR2GRAY)
    _image = __image.astype(np.float32)/255.0
    X.append(_image)
    Y.append([encode_maps[c] for c in labels[i]] + [2])





train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size = 0.05)



print([decode_maps[c] for c in Y[-1]])




class AttentionMechanism(object):
    """Class to compute attention over an image"""

    def __init__(self, img, dim_e, tiles=1):
        """Stores the image under the right shape.
        We loose the H, W dimensions and merge them into a single
        dimension that corresponds to "regions" of the image.
        Args:
            img: (tf.Tensor) image
            dim_e: (int) dimension of the intermediary vector used to
                compute attention
            tiles: (int) default 1, input to context h may have size
                    (tile * batch_size, ...)
        """
        if len(img.shape) == 3:
            self._img = img
        elif len(img.shape) == 4:
            N    = tf.shape(img)[0]
            H, W = tf.shape(img)[1], tf.shape(img)[2] # image
            C    = img.shape[3].value                 # channels
            self._img = tf.reshape(img, shape=[N, H*W, C])
        else:
            print("Image shape not supported")
            raise NotImplementedError

        # dimensions
        self._n_regions  = tf.shape(self._img)[1]
        self._n_channels = self._img.shape[2].value
        self._dim_e      = dim_e
        self._tiles      = tiles
        self._scope_name = "att_mechanism"

        # attention vector over the image
        self._att_img = tf.layers.dense(
            inputs=self._img,
            units=self._dim_e,
            use_bias=False,
            name="att_img")


    def context(self, h):
        """Computes attention
        Args:
            h: (batch_size, num_units) hidden state
        Returns:
            c: (batch_size, channels) context vector
        """
        with tf.variable_scope(self._scope_name):
            if self._tiles > 1:
                att_img = tf.expand_dims(self._att_img, axis=1)
                att_img = tf.tile(att_img, multiples=[1, self._tiles, 1, 1])
                att_img = tf.reshape(att_img, shape=[-1, self._n_regions,
                        self._dim_e])
                img = tf.expand_dims(self._img, axis=1)
                img = tf.tile(img, multiples=[1, self._tiles, 1, 1])
                img = tf.reshape(img, shape=[-1, self._n_regions,
                        self._n_channels])
            else:
                att_img = self._att_img
                img     = self._img

            # computes attention over the hidden vector
            att_h = tf.layers.dense(inputs=h, units=self._dim_e, use_bias=False)

            # sums the two contributions
            att_h = tf.expand_dims(att_h, axis=1)
            att = tf.tanh(att_img + att_h)

            # computes scalar product with beta vector
            # works faster with a matmul than with a * and a tf.reduce_sum
            att_beta = tf.get_variable("att_beta", shape=[self._dim_e, 1],
                    dtype=tf.float32)
            att_flat = tf.reshape(att, shape=[-1, self._dim_e])
            e = tf.matmul(att_flat, att_beta)
            e = tf.reshape(e, shape=[-1, self._n_regions])

            # compute weights
            a = tf.nn.softmax(e)
            a = tf.expand_dims(a, axis=-1)
            c = tf.reduce_sum(a * img, axis=1)

            return c


    def initial_cell_state(self, cell):
        """Returns initial state of a cell computed from the image
        Assumes cell.state_type is an instance of named_tuple.
        Ex: LSTMStateTuple
        Args:
            cell: (instance of RNNCell) must define _state_size
        """
        _states_0 = []
        for hidden_name in cell._state_size._fields:
            hidden_dim = getattr(cell._state_size, hidden_name)
            h = self.initial_state(hidden_name, hidden_dim)
            _states_0.append(h)

        initial_state_cell = type(cell.state_size)(*_states_0)

        return initial_state_cell


    def initial_state(self, name, dim):
        """Returns initial state of dimension specified by dim"""
        with tf.variable_scope(self._scope_name):
            img_mean = tf.reduce_mean(self._img, axis=1)
            W = tf.get_variable("W_{}_0".format(name), shape=[self._n_channels,
                    dim])
            b = tf.get_variable("b_{}_0".format(name), shape=[dim])
            h = tf.tanh(tf.matmul(img_mean, W) + b)

            return h




AttentionState = collections.namedtuple("AttentionState", ("cell_state", "o"))


class AttentionCell(RNNCell):
    def __init__(self, cell, attention_mechanism, dropout, dim_e,
                 dim_o, num_units,
        num_proj, dtype=tf.float32):
        """
        Args:
            cell: (RNNCell)
            attention_mechanism: (AttentionMechanism)
            dropout: (tf.float)
            attn_cell_config: (dict) hyper params
        """
        # variables and tensors
        self._cell                = cell
        self._attention_mechanism = attention_mechanism
        self._dropout             = dropout

        # hyperparameters and shapes
        self._n_channels     = self._attention_mechanism._n_channels
        self._dim_e          = dim_e
        self._dim_o          = dim_o
        self._num_units      = num_units
        self._num_proj       = num_proj
        self._dtype          = dtype

        # for RNNCell
        self._state_size = AttentionState(self._cell._state_size, self._dim_o)


    @property
    def state_size(self):
        return self._state_size


    @property
    def output_size(self):
        return self._num_proj


    @property
    def output_dtype(self):
        return self._dtype


    def initial_state(self):
        """Returns initial state for the lstm"""
        initial_cell_state = self._attention_mechanism.initial_cell_state(self._cell)
        initial_o          = self._attention_mechanism.initial_state("o", self._dim_o)

        return AttentionState(initial_cell_state, initial_o)


    def step(self, embedding, attn_cell_state):
        """
        Args:
            embedding: shape = (batch_size, dim_embeddings) embeddings
                from previous time step
            attn_cell_state: (AttentionState) state from previous time step
        """
        prev_cell_state, o = attn_cell_state

        scope = tf.get_variable_scope()
        with tf.variable_scope(scope):
            # compute new h
            x                     = tf.concat([embedding, o], axis=-1)
            new_h, new_cell_state = self._cell.__call__(x, prev_cell_state)
            new_h = tf.nn.dropout(new_h, self._dropout)

            # compute attention
            c = self._attention_mechanism.context(new_h)

            # compute o
            o_W_c = tf.get_variable("o_W_c", dtype=tf.float32,
                    shape=(self._n_channels, self._dim_o))
            o_W_h = tf.get_variable("o_W_h", dtype=tf.float32,
                    shape=(self._num_units, self._dim_o))

            new_o = tf.tanh(tf.matmul(new_h, o_W_h) + tf.matmul(c, o_W_c))
            new_o = tf.nn.dropout(new_o, self._dropout)

            y_W_o = tf.get_variable("y_W_o", dtype=tf.float32,
                    shape=(self._dim_o, self._num_proj))
            logits = tf.matmul(new_o, y_W_o)

            # new Attn cell state
            new_state = AttentionState(new_cell_state, new_o)

            return logits, new_state


    def __call__(self, inputs, state):
        """
        Args:
            inputs: the embedding of the previous word for training only
            state: (AttentionState) (h, o) where h is the hidden state and
                o is the vector used to make the prediction of
                the previous word
        """
        new_output, new_state = self.step(inputs, state)

        return (new_output, new_state)


def add_timing_signal_nd(x, min_timescale=1.0, max_timescale=1.0e4):
    """Adds a bunch of sinusoids of different frequencies to a Tensor.

    Each channel of the input Tensor is incremented by a sinusoid of a difft
    frequency and phase in one of the positional dimensions.

    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.

    The use of relative position is possible because sin(a+b) and cos(a+b) can
    be experessed in terms of b, sin(a) and cos(a).

    x is a Tensor with n "positional" dimensions, e.g. one dimension for a
    sequence or two dimensions for an image

    We use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels // (n * 2). For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.

    Args:
        x: a Tensor with shape [batch, d1 ... dn, channels]
        min_timescale: a float
        max_timescale: a float

    Returns:
        a Tensor the same shape as x.

    """
    static_shape = x.get_shape().as_list()
    num_dims = len(static_shape) - 2
    channels = tf.shape(x)[-1]
    num_timescales = channels // (num_dims * 2)
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
            tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    for dim in xrange(num_dims):
        length = tf.shape(x)[dim + 1]
        position = tf.to_float(tf.range(length))
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(
                inv_timescales, 0)
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        prepad = dim * 2 * num_timescales
        postpad = channels - (dim + 1) * 2 * num_timescales
        signal = tf.pad(signal, [[0, 0], [prepad, postpad]])
        for _ in xrange(1 + dim):
            signal = tf.expand_dims(signal, 0)
        for _ in xrange(num_dims - 1 - dim):
            signal = tf.expand_dims(signal, -2)
        x += signal
    return x




attention_size = 256
size_layer = 256
embedded_size = 256
beam_width = 15
learning_rate = 1e-4


# tf.contrib.seq2seq as decoder part

class Model:
    def __init__(self):
        self.X = tf.placeholder(tf.float32, shape=(None, 32, 480, 1))
        self.Y = tf.placeholder(tf.int32, [None, None])
        self.Y_seq_len = tf.count_nonzero(self.Y, 1, dtype=tf.int32)
        batch_size = tf.shape(self.X)[0]
        x_len = tf.shape(self.X)[2] // 2
        main = tf.strided_slice(self.Y, [0, 0], [batch_size, -1], [1, 1])
        decoder_input = tf.concat([tf.fill([batch_size, 1], GO), main], 1)
        
        decoder_embeddings = tf.Variable(tf.random_uniform([len(encode_maps), embedded_size], -1, 1))
        
        img = self.X
        
        out = tf.layers.conv2d(img, 64, 3, 1, "SAME",
                activation=tf.nn.relu)
        out = tf.layers.max_pooling2d(out, 2, 2, "SAME")

        out = tf.layers.conv2d(out, 128, 3, 1, "SAME",
                activation=tf.nn.relu)
        out = tf.layers.max_pooling2d(out, 2, 2, "SAME")

        out = tf.layers.conv2d(out, 256, 3, 1, "SAME",
                activation=tf.nn.relu)

        out = tf.layers.conv2d(out, 256, 3, 1, "SAME",
                activation=tf.nn.relu)
        out = tf.layers.max_pooling2d(out, (2, 1), (2, 1), "SAME")
        out = tf.layers.conv2d(out, 512, 3, 1, "SAME",
                activation=tf.nn.relu)
        out = tf.layers.max_pooling2d(out, (1, 2), (1, 2), "SAME")
        out = tf.layers.conv2d(out, 512, 3, 1, "VALID",
                activation=tf.nn.relu)
        img = add_timing_signal_nd(out)
        print(img)
        
        with tf.variable_scope("attn_cell", reuse=False):
            attn_meca = AttentionMechanism(img, attention_size)
            recu_cell = tf.nn.rnn_cell.LSTMCell(size_layer)
            attn_cell = AttentionCell(recu_cell, attn_meca, 1.0,
                        attention_size, attention_size, size_layer, len(encode_maps))

            encoder_state = attn_cell.initial_state()

            training_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
                    inputs = tf.nn.embedding_lookup(decoder_embeddings, decoder_input),
                    sequence_length = self.Y_seq_len,
                    embedding = decoder_embeddings,
                    sampling_probability = 0.5,
                    time_major = False)
            training_decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell = attn_cell,
                    helper = training_helper,
                    initial_state = encoder_state,
                    output_layer = None)
            training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder = training_decoder,
                    impute_finished = True,
                    maximum_iterations = tf.reduce_max(self.Y_seq_len))
        
        with tf.variable_scope("attn_cell", reuse=True):
            attn_meca = AttentionMechanism(img, attention_size, tiles=beam_width)
            recu_cell = tf.nn.rnn_cell.LSTMCell(size_layer, reuse = True)
            attn_cell = AttentionCell(recu_cell, attn_meca, 1.0,
                        attention_size, attention_size, size_layer, len(encode_maps))
            
            encoder_state = attn_cell.initial_state()
            
            predicting_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell = attn_cell,
                embedding = decoder_embeddings,
                start_tokens = tf.tile(tf.constant([GO], dtype=tf.int32), [batch_size]),
                end_token = EOS,
                initial_state = tf.contrib.seq2seq.tile_batch(encoder_state, beam_width),
                beam_width = beam_width,
                output_layer = None,
                length_penalty_weight = 0.0)
            predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder = predicting_decoder,
                impute_finished = False,
                maximum_iterations = x_len)
            
        self.training_logits = training_decoder_output.rnn_output
        self.predicting_ids = predicting_decoder_output.predicted_ids
        
        masks = tf.sequence_mask(self.Y_seq_len, tf.reduce_max(self.Y_seq_len), dtype=tf.float32)
        self.cost = tf.contrib.seq2seq.sequence_loss(logits = self.training_logits,
                                                     targets = self.Y,
                                                     weights = masks)
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
        y_t = tf.argmax(self.training_logits,axis=2)
        y_t = tf.cast(y_t, tf.int32)
        self.prediction = tf.boolean_mask(y_t, masks)
        mask_label = tf.boolean_mask(self.Y, masks)
        correct_pred = tf.equal(self.prediction, mask_label)
        correct_index = tf.cast(correct_pred, tf.float32)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tf.reset_default_graph()
# sess = tf.InteractiveSession()
sess = tf.Session()
model = Model()
sess.run(tf.global_variables_initializer())


def pad_sentence_batch(sentence_batch, pad_int):
    padded_seqs = []
    seq_lens = []
    max_sentence_len = max([len(sentence) for sentence in sentence_batch])
    for sentence in sentence_batch:
        padded_seqs.append(sentence + [pad_int] * (max_sentence_len - len(sentence)))
        seq_lens.append(len(sentence))
    return padded_seqs, seq_lens


batch_x = train_X[:5]
batch_x = np.array(batch_x).reshape((len(batch_x), image_height, image_width,image_channel))
y = train_Y[:5]
batch_y, _ = pad_sentence_batch(y, 0)
loss, logits, acc = sess.run([model.cost, model.training_logits, model.accuracy], feed_dict = {model.X: batch_x,
                                                          model.Y: batch_y})


saver_all = tf.train.Saver(tf.all_variables())

checkpoint_path = os.path.join('checkpoints', "model.ckpt")

ckpt = tf.train.get_checkpoint_state('checkpoints')
if ckpt:
    print('restore checkpoints')
    saver_all.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.initialize_all_variables())

for e in range(epoch):
    pbar = tqdm(
        range(0, len(train_X), batch_size), desc = 'minibatch loop')
    train_loss, train_acc, test_loss, test_acc = [], [], [], []
    for i in pbar:
        index = min(i + batch_size, len(train_X))
        batch_x = train_X[i : index]
        batch_x = np.array(batch_x).reshape((len(batch_x), image_height, image_width,image_channel))
        y = train_Y[i : index]
        batch_y, _ = pad_sentence_batch(y, 0)
        feed = {model.X: batch_x,
                model.Y: batch_y}
        accuracy, loss, _ = sess.run([model.accuracy,model.cost,model.optimizer],
                                    feed_dict = feed)
        train_loss.append(loss)
        train_acc.append(accuracy)
        pbar.set_postfix(cost = loss, accuracy = accuracy)
    
    
    pbar = tqdm(
        range(0, len(test_X), batch_size), desc = 'minibatch loop')
    for i in pbar:
        index = min(i + batch_size, len(test_X))
        batch_x = test_X[i : index]
        batch_x = np.array(batch_x).reshape((len(batch_x), image_height, image_width,image_channel))
        y = test_Y[i : index]
        batch_y, _ = pad_sentence_batch(y, 0)
        feed = {model.X: batch_x,
                model.Y: batch_y,}
        accuracy, loss = sess.run([model.accuracy,model.cost],
                                    feed_dict = feed)

        test_loss.append(loss)
        test_acc.append(accuracy)
        pbar.set_postfix(cost = loss, accuracy = accuracy)
    saver_all.save(sess, checkpoint_path, global_step=e)
    
    print('epoch %d, training avg loss %f, training avg acc %f'%(e+1,
                                                                 np.mean(train_loss),np.mean(train_acc)))
    print('epoch %d, testing avg loss %f, testing avg acc %f'%(e+1,
                                                              np.mean(test_loss),np.mean(test_acc)))


# decoded = sess.run(model.predicting_ids, feed_dict = {model.X: batch_x[:1],
#                                           model.Y: batch_y[:1]})[0]
# print(decoded.shape)

# for i in range(decoded.shape[1]):
#     d = decoded[:,0]
#     print(''.join([decode_maps[i] for i in d if i not in [0,1,2]]))
