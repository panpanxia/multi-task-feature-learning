# Objective : @TODO
# Created by: ece
# Created on: 27.03.2018

"""Builds the MTFL network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

import mtfl_input
import datasets

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch""")
tf.app.flags.DEFINE_string('data_dir', 'data',
                           """Path to mtfl directory""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16""")

# Global constants describing the MTFL data set.
IMAGE_SIZE = mtfl_input.IMAGE_SIZE
NUM_CLASSES = mtfl_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = mtfl_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_TEST = mtfl_input.NUM_EXAMPLES_PER_EPOCH_FOR_TEST

# Constants describing the training process
MOVING_AVERAGE_DECAY = 0.999        # The decay to use for moving average
NUM_EPOCHS_PER_DECAY = 350.0        # Epochs after which learning rate decays
LEARNING_RATE_DECAY_FACTOR = 0.1    # Learning rate decay factor
INITIAL_LEARNING_RATE = 0.1         # Initial learning rate


TOWER_NAME = 'tower'


def _activation_summary(x):
    """
    Helpers to create summaries for activations.

    Creates a summary that provides a histogram of activations
    Creates a summary that measures the sparsity of activations
    :param x: Tensor
    :return: nothing
    """
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity',
                      tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
    """ Helper tp create a variable stored in CPU memory.
    Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for the variable
    Returns:
        Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """ Helper to create an initialized Variable with weight decay.
    Note that the variable is initialized with a truncated normal distributions.
    A weight decay is added only if one is specified.

    Args:
        name: name of the variable
        shape: list of ints
        sttdev: standart deviation of a truncated Gaussian
        wd: add L2loss weight decay multiplied by this float If none,
        weight decay is not added for this variable.
    Returns:
        Variable tensor
    """
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def distorted_inputs():
    """
    Construct distorted input for MTFL training use Reader ops.

    :return: Images. 4D Tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
             Labels. 1D Tensor of [batch_size] size.
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data dir')

    data_set_tag = os.path.split(FLAGS.bin_path)[1]
    data_set = datasets.bring(data_set_tag)

    images, labels = mtfl_input.distorted_inputs(filenames=data_set.files, batch_size=FLAGS.batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels


def train_inputs():
    """
    Construct distorted input for MTFL training use Reader ops.
    Use it when you use train/val split.
    I got to write it because, you can not feed a session with a tensor
    :return: Images. 4D list of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
             Labels. 1D list of [batch_size] size.
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data dir')

    data_set_tag = os.path.split(FLAGS.bin_path)[1]
    data_set = datasets.bring(data_set_tag)

    images, labels = mtfl_input.train_inputs(filenames=data_set.files, batch_size=FLAGS.batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels


def validation_inputs():
    """
    Construct validation input for MTFL
    :return: Images. 4D Tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
             Labels. 1D Tensor of [batch_size] size.
    """
    data_set_tag = os.path.split(FLAGS.bin_path)[1]
    data_set = datasets.bring(data_set_tag)

    images, labels = mtfl_input.validation_inputs(filenames=data_set.files, batch_size=FLAGS.batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels


def _add_loss_summaries(total_loss):
    """
    Add summaries for losses in MTFL model.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    :param total_loss: Total loss from loss()
    :return: loss averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def train(total_loss, global_step):
    """
    Train MTFL model.

    Create an optimizer and apply to all trainable variables.
    Add moving average for all trainable variables.
    :param total_loss: Total loss from loss().
    :param global_step: Integer variable counting the number of training steps processed.
    :return: train_op for training
    """
    # Variables that affect learning rate
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute the gradients
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients:
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables:
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def inference(images):
    """
    Build MTFL model.
    :param images: Images returned from distorted_inputs()
    :return: Logits
    """

    # We instantiate all variables using tf.get_variable() instead
    # of tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable()

    # Convolution 1
    with tf.variable_scope('convolution-1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 1, 16],
                                             stddev=5e-2,
                                             wd=None)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding="VALID")
        biases = _variable_on_cpu('biases', [16], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        # activation summary here
        _activation_summary(conv1)

    # Max pool 1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='VALID', name='max-pool-1')
    # # Normalization 1
    # norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
    #                   name='norm1')

    # Convolution 2
    with tf.variable_scope('convolution-2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 16, 48],
                                             stddev=5e-2,
                                             wd=None)
        # conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='VALID')
        biases = _variable_on_cpu('biases', [48], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
        # activation summary here
        _activation_summary(conv2)

    # norm 2
    # norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
    #                   name='norm2')
    # pool 2
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='VALID', name='pool2')

    # Convolution 3
    with tf.variable_scope('convolution-3') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 48, 64],
                                             stddev=5e-2,
                                             wd=None)
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='VALID')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(pre_activation, name=scope.name)
        # activation summary here
        _activation_summary(conv3)

    # norm 3
    # norm3 = tf.nn.lrn(conv3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
    #                   name='norm2')
    # pool 3
    pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='VALID', name='pool3')

    # Convolution 4
    with tf.variable_scope('convolution-4') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[2, 2, 64, 64],
                                             stddev=5e-2,
                                             wd=None)
        conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='VALID')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(pre_activation, name=scope.name)
        # activation summary here
        _activation_summary(conv4)

    # Fully Connect: Flatten
    with tf.variable_scope('fully-connect') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(conv4, [FLAGS.batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[dim, 100],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [100], tf.constant_initializer(0.1))
        fully_connect = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        # Activation summary here
        _activation_summary(fully_connect)

    # Softmax classifier here:
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [100, mtfl_input.NUM_CLASSES],
                                              stddev=1/512.0, wd=None)
        biases = _variable_on_cpu('biases', [mtfl_input.NUM_CLASSES],
                                  tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(fully_connect, weights), biases, name=scope.name)
        # Activation summary here
        _activation_summary(softmax_linear)
    return softmax_linear


def classification_loss(logits, labels):
    """
    Add L2 loss to all trainable variables.
    Add summary for loss and loss/avg
    :param logits: from inference
    :param labels: labels from distorted_inputs(). 1-d tensor of shape [batch_size]
    :return: Loss tensor of type float
    """
    # Calculate the average cross entropy loss accross the batch
    labels = tf.cast(labels, tf.int64)
    labels = tf.reshape(labels, [-1])
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')

    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def train(total_loss, global_step):
    """
    Train MTFL model.
    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.
    :param total_loss: total_loss from loss()
    :param global_step: Integer Variable counting the number of training steps
      processed.
    :return: train_op: op for training.
    """

    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)

    with tf.control_dependencies([apply_gradient_op]):
      variables_averages_op = variable_averages.apply(tf.trainable_variables())

    return variables_averages_op


def maybe_make_bins():
    """
    Read the images folder and make the images as bins
    in order to make whole process a lot faster
    and in order to model act as if cifar 10 cnn tf model.
    :return:
    """
    datasets.write_to_binary(FLAGS.data_set)