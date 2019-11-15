import numpy as np
import os
import tensorflow as tf
from skimage.transform import resize as imresize
import cv2
import time
import csv
from tqdm import tqdm
from sklearn.model_selection import train_test_split



os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def add_padding(image, max_width = 480):
    h, w = image.shape[:2]
    right = max_width - w
    image = cv2.copyMakeBorder(image,0,0,0,right,cv2.BORDER_CONSTANT,value=(255,255,255))
    return image

def resize_scaled(image, height = 32):
    h, w = image.shape[:2]
    new_width = int((height * w)/h)
    image = cv2.resize(image,(new_width,height),interpolation=cv2.INTER_AREA)
    return image

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
    # random.shuffle(output)
    return output


# directory = 'ocrv4_dataset'
directory = 'ocrv4_dataset'
images = []
# images = []
labels = []
data = read_dataset(f'{directory}/annot_cleaned.csv')
# for line in data:
#     image_path = f'{directory}/{line[0]}'
#     text = line[1]
#     images.append(image_path)
#     labels.append(text)
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
for i, char in enumerate(charset, 1):
    encode_maps[char] = i
    decode_maps[i] = char

SPACE_INDEX = 0
SPACE_TOKEN = ''
encode_maps[SPACE_TOKEN] = SPACE_INDEX
decode_maps[SPACE_INDEX] = SPACE_TOKEN


image_height = 32
image_width = 480
image_channel = 1
max_stepsize = 128
num_hidden = 256
epoch = 20
batch_size = 16
initial_learning_rate = 1e-3


def pad_second_dim(x, desired_size):
    padding = tf.tile([[0]], tf.stack([tf.shape(x)[0], desired_size - tf.shape(x)[1]], 0))
    return tf.concat([x, padding], 1)

class Model:
    def __init__(self):
        self.X = tf.placeholder(tf.float32, [None, image_height, image_width, image_channel])
        self.Y = tf.sparse_placeholder(tf.int32)
        self.SEQ_LEN = tf.placeholder(tf.int32, [None])
        self.label = tf.placeholder(tf.int32, [None, None])
        self.Y_seq_len = tf.placeholder(tf.int32, [None])
        batch_size = tf.shape(self.X)[0]
        filters = [64, 128, 128, max_stepsize]
        strides = [1, 2]
        x = self.conv2d(self.X, 'cnn-1', 3, 1, filters[0], strides[0])
        x = self.batch_norm('bn1', x)
        x = self.leaky_relu(x, 0.01)
        x = self.max_pool(x, 2, strides[1])
        x = self.conv2d(x, 'cnn-2', 3, filters[0], filters[1], strides[0])
        x = self.batch_norm('bn2', x)
        x = self.leaky_relu(x, 0.01)
        x = self.max_pool(x, 2, strides[1])
        x = self.conv2d(x, 'cnn-3', 3, filters[1], filters[2], strides[0])
        x = self.batch_norm('bn3', x)
        x = self.leaky_relu(x, 0.01)
        x = self.max_pool(x, 2, strides[1])
        x = self.conv2d(x, 'cnn-4', 3, filters[2], filters[3], strides[0])
        x = self.batch_norm('bn4', x)
        x = self.leaky_relu(x, 0.01)
        x = self.max_pool(x, 2, strides[1])
        x = tf.reshape(x, [batch_size, -1, filters[3]])
        x = tf.transpose(x, [0, 2, 1])
        x = tf.reshape(x, [batch_size, filters[3], 4 * 15])
        # cell = tf.contrib.rnn.LSTMCell(num_hidden)
        # cell1 = tf.contrib.rnn.LSTMCell(num_hidden)
        # stack = tf.contrib.rnn.MultiRNNCell([cell, cell1])
        # outputs, _ = tf.nn.dynamic_rnn(stack, x, self.SEQ_LEN, dtype=tf.float32)
        outputs = tf.nn.cudnn_rnn.CudnnGRU(2,num_hidden,direction='bidirectional')
        outputs = tf.reshape(outputs, [-1, num_hidden])
        self.logits = tf.layers.dense(outputs, num_classes)
        shape = tf.shape(x)
        self.logits = tf.reshape(self.logits, [shape[0], -1, num_classes])
        self.logits = tf.transpose(self.logits, (1, 0, 2))
        self.global_step = tf.Variable(0, trainable=False)
        self.loss = tf.nn.ctc_loss(labels=self.Y,
                                   inputs=self.logits,
                                   sequence_length=self.SEQ_LEN)
        self.cost = tf.reduce_mean(self.loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=initial_learning_rate).minimize(self.cost)
        self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(self.logits,
                                                                    self.SEQ_LEN,
                                                                    merge_repeated=False)
        decoded = tf.to_int32(self.decoded[0])
        self.dense_decoded = tf.sparse_tensor_to_dense(decoded)
        
        preds = self.dense_decoded[:, :tf.reduce_max(self.Y_seq_len)]
        masks = tf.sequence_mask(self.Y_seq_len, tf.reduce_max(self.Y_seq_len), dtype=tf.float32)
        preds = pad_second_dim(preds, tf.reduce_max(self.Y_seq_len))
        y_t = tf.cast(preds, tf.int32)
        self.prediction = tf.boolean_mask(y_t, masks)
        mask_label = tf.boolean_mask(self.label, masks)
        self.mask_label = mask_label
        correct_pred = tf.equal(self.prediction, mask_label)
        correct_index = tf.cast(correct_pred, tf.float32)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
        
    def conv2d(self, x, name, filter_size, channel_in, channel_out, strides):
        with tf.variable_scope(name):
            return tf.layers.conv2d(x, channel_out, filter_size, strides, padding='SAME')
        
    
    def batch_norm(self, name, x):
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]
            beta = tf.get_variable('beta', params_shape, tf.float32,
                                   initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable('gamma', params_shape, tf.float32,
                                    initializer=tf.constant_initializer(1.0, tf.float32))
            mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')
            x_bn = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
            x_bn.set_shape(x.get_shape())
            return x_bn
        
    def leaky_relu(self, x, leak=0):
        return tf.where(tf.less(x, 0.0), leak * x, x, name='leaky_relu')
    
    def max_pool(self, x, size, strides):
        return tf.nn.max_pool(x, 
                              ksize=[1, size, size, 1],
                              strides=[1, strides, strides, 1],
                              padding='SAME',
                              name='max_pool')


def sparse_tuple_from_label(sequences, dtype=np.int32):
    indices, values = [], []
    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
    return indices, values, shape


def pad_sentence_batch(sentence_batch, pad_int):
    padded_seqs = []
    seq_lens = []
    max_sentence_len = max([len(sentence) for sentence in sentence_batch])
    for sentence in sentence_batch:
        padded_seqs.append(sentence + [pad_int] * (max_sentence_len - len(sentence)))
        seq_lens.append(len(sentence))
    return padded_seqs, seq_lens


tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Model()
sess.run(tf.global_variables_initializer())




X, Y = [], []
for i in tqdm(range(len(images))):
    # img = images[i]
    # X.append(imresize(cv2.imread(directory+img, 0).astype(np.float32)/255., (image_height,image_width)))
    img = images[i]
    __image = cv2.imread(img)
    if __image is not None:
        # __image = resize_scaled(__image)
        __image = cv2.cvtColor(__image,cv2.COLOR_BGR2GRAY)
        _image = __image.astype(np.float32)
        _image = _image / 255.0
        X.append(_image)
        Y.append([SPACE_INDEX if labels[0] == SPACE_TOKEN else encode_maps[c] for c in labels[i]])



train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size = 0.05)

saver_all = tf.train.Saver(tf.all_variables())

checkpoint_path = os.path.join('checkpoints_ctc', "model.ckpt")

ckpt = tf.train.get_checkpoint_state('checkpoints_ctc')
if ckpt:
    print('restore checkpoints')
    saver_all.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.initialize_all_variables())

for e in range(epoch):
    pbar = tqdm(
        range(0, len(train_X), batch_size), desc = 'minibatch loop')
    train_loss, train_acc, test_loss, test_acc = [], [], [], []
    total_lost, total_acc = 0, 0
    for i in pbar:
        index = min(i + batch_size, len(train_X))
        batch_x = train_X[i : index]
        batch_x = np.array(batch_x).reshape((len(batch_x), image_height, image_width,image_channel))
        y = train_Y[i : index]
        batch_y = sparse_tuple_from_label(y)
        batch_label, batch_length = pad_sentence_batch(y, 0)
        batch_len = np.asarray([max_stepsize for _ in [1]*len(batch_x)], dtype=np.int64)
        feed = {model.X: batch_x,
                model.Y: batch_y,
                model.SEQ_LEN: batch_len,
               model.label: batch_label,
               model.Y_seq_len: batch_length}
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
        batch_y = sparse_tuple_from_label(y)
        batch_label, batch_length = pad_sentence_batch(y, 0)
        batch_len = np.asarray([max_stepsize for _ in [1]*len(batch_x)], dtype=np.int64)
        feed = {model.X: batch_x,
                model.Y: batch_y,
                model.SEQ_LEN: batch_len,
               model.label: batch_label,
               model.Y_seq_len: batch_length}
        accuracy, loss = sess.run([model.accuracy,model.cost], feed_dict = feed)

        test_loss.append(loss)
        test_acc.append(accuracy)
        pbar.set_postfix(cost = loss, accuracy = accuracy)
    
    print('epoch %d, training avg loss %f, training avg acc %f'%(e+1,
                                                                 np.mean(train_loss),np.mean(train_acc)))
    print('epoch %d, testing avg loss %f, testing avg acc %f'%(e+1,
                                                              np.mean(test_loss),np.mean(test_acc)))


# decoded = sess.run(model.dense_decoded, feed_dict = {model.X: batch_x[:1],
#                                           model.SEQ_LEN: batch_len[:1]})