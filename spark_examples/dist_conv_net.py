from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys

import tensorflow.python.platform

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import itertools
import pandas as pd
from tensorflow.examples.tutorials.mnist import input_data
from pyspark.context import SparkContext

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = 'data'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 5000  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPOCHS = 2

mnist = input_data.read_data_sets("/tmp/MNIST_data/", one_hot=True)

# Initialize Spark Context
sc = SparkContext()

def maybe_download(filename):
  """Download the data from Yann's website, unless it's already here."""
  if not os.path.exists(WORK_DIRECTORY):
    os.mkdir(WORK_DIRECTORY)
  filepath = os.path.join(WORK_DIRECTORY, filename)
  if not os.path.exists(filepath):
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
  return filepath


def extract_data(filename, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].

  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
    return data


def extract_labels(filename, num_images):
  """Extract the labels into a 1-hot matrix [image index, label index]."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = np.frombuffer(buf, dtype=np.uint8)
  # Convert to dense 1-hot representation.
  return (np.arange(NUM_LABELS) == labels[:, None]).astype(np.float32)


def fake_data(num_images):
  """Generate a fake dataset that matches the dimensions of MNIST."""
  data = np.ndarray(
      shape=(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
      dtype=np.float32)
  labels = np.zeros(shape=(num_images, NUM_LABELS), dtype=np.float32)
  for image in xrange(num_images):
    label = image % 2
    data[image, :, :, 0] = label - 0.5
    labels[image, label] = 1.0
  return data, labels


def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and 1-hot labels."""
  return 100.0 - (
      100.0 *
      np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) /
      predictions.shape[0])

train_data_filename = maybe_download('/tmp/MNIST_data/train-images-idx3-ubyte.gz')
train_labels_filename = maybe_download('/tmp/MNIST_data/train-labels-idx1-ubyte.gz')
test_data_filename = maybe_download('/tmp/MNIST_data/t10k-images-idx3-ubyte.gz')
test_labels_filename = maybe_download('/tmp/MNIST_data/t10k-labels-idx1-ubyte.gz')

train_data_filename = '/tmp/MNIST_data/train-images-idx3-ubyte.gz'
train_labels_filename = '/tmp/MNIST_data/train-labels-idx1-ubyte.gz'
test_data_filename = '/tmp/MNIST_data/t10k-images-idx3-ubyte.gz'
test_labels_filename = '/tmp/MNIST_data/t10k-labels-idx1-ubyte.gz'

train_data = extract_data(train_data_filename, 60000)
train_labels = extract_labels(train_labels_filename, 60000)
test_data = extract_data(test_data_filename, 10000)
test_labels = extract_labels(test_labels_filename, 10000)

validation_data = train_data[:VALIDATION_SIZE, :, :, :]
validation_labels = train_labels[:VALIDATION_SIZE]
train_data = train_data[VALIDATION_SIZE:, :, :, :]
train_labels = train_labels[VALIDATION_SIZE:]
num_epochs = NUM_EPOCHS
train_size = train_labels.shape[0]

class ConvNet(object): pass

def create_graph(base_learning_rate = 0.01, decay_rate = 0.95, conv1_size=32, conv2_size=64, fc1_size=512):
    train_data_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    train_labels_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_LABELS))
    validation_data_node = tf.constant(validation_data)
    test_data_node = tf.constant(test_data)
    conv1_weights = tf.Variable( tf.truncated_normal([5, 5, NUM_CHANNELS, conv1_size],stddev=0.1, seed=SEED))
    conv1_biases = tf.Variable(tf.zeros([conv1_size]))
    conv2_weights = tf.Variable(tf.truncated_normal([5, 5, conv1_size, conv2_size], stddev=0.1,seed=SEED))
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[conv2_size]))
    fc1_weights = tf.Variable(tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * conv2_size, fc1_size],stddev=0.1,seed=SEED))
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[fc1_size]))
    fc2_weights = tf.Variable(tf.truncated_normal([fc1_size, NUM_LABELS],stddev=0.1,seed=SEED))
    fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))

    def model(data, train=False):
        conv = tf.nn.conv2d(data, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
        pool = tf.nn.max_pool(relu,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')
        conv = tf.nn.conv2d(pool,conv2_weights,strides=[1, 1, 1, 1],padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
        pool = tf.nn.max_pool(relu,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')
        pool_shape = pool.get_shape().as_list()
        reshape = tf.reshape(pool, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        if train:
          hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
        return tf.matmul(hidden, fc2_weights) + fc2_biases

    logits = model(train_data_node, True)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, train_labels_node))
    regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) + tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
    loss += 5e-4 * regularizers
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(base_learning_rate, batch * BATCH_SIZE, train_size, decay_rate, staircase=True)
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=batch)
    train_prediction = tf.nn.softmax(logits)
    validation_prediction = tf.nn.softmax(model(validation_data_node))
    test_prediction = tf.nn.softmax(model(test_data_node))
    res = ConvNet()
    res.train_prediction = train_prediction
    res.optimizer = optimizer
    res.loss = loss
    res.learning_rate = learning_rate
    res.validation_prediction = validation_prediction
    res.test_prediction = test_prediction
    res.train_data_node = train_data_node
    res.train_labels_node = train_labels_node

    return res

# Broadcast data to Spark workers
train_data_bc = sc.broadcast(train_data)
train_labels_bc = sc.broadcast(train_labels)

def run(base_learning_rate, decay_rate, fc1_size):
  train_data = train_data_bc.value
  train_labels = train_labels_bc.value
  res = {}
  res['base_learning_rate'] = base_learning_rate
  res['decay_rate'] = decay_rate
  res['fc1_size'] = fc1_size
  res['minibatch_loss'] = 100.0
  res['test_error'] = 100.0
  res['validation_error'] = 100.0
  try:
    with tf.Session() as s:
      graph = create_graph(base_learning_rate, decay_rate, fc1_size=fc1_size)
      tf.initialize_all_variables().run()
      for step in xrange(num_epochs * train_size // BATCH_SIZE):
        offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
        batch_data = train_data[offset:(offset + BATCH_SIZE), :, :, :]
        batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
        feed_dict = {graph.train_data_node: batch_data,
                     graph.train_labels_node: batch_labels}
        _, l, lr, predictions = s.run([graph.optimizer,
                                       graph.loss,
                                       graph.learning_rate,
                                       graph.train_prediction],
                                      feed_dict=feed_dict)
        res['minibatch_loss'] = l
      res['test_error'] = error_rate(graph.test_prediction.eval(), test_labels)
      res['validation_error'] = error_rate(graph.validation_prediction.eval(), validation_labels)
      return res
  except Exception as e:
    print("Something fucked up.\n \n", e)
  return res

base_learning_rates = [float(x) for x in np.logspace(-3, -1, num=10, base=10.0)]
decay_rates = [0.95]
fc1_sizes = [64, 128, 256, 512, 1024]
all_experiments = list(itertools.product(base_learning_rates, decay_rates, fc1_sizes))
print(len(all_experiments))

all_exps_rdd = sc.parallelize(all_experiments, numSlices=len(all_experiments))

num_nodes = 4
n = max(2, int(len(all_experiments) // num_nodes))
grouped_experiments = [all_experiments[i:i+n] for i in range(0, len(all_experiments), n)]
all_exps_rdd = sc.parallelize(grouped_experiments, numSlices=len(grouped_experiments))
results = all_exps_rdd.flatMap(lambda z: [run(*y) for y in z]).collect()

df = pd.DataFrame(results)

print(df)