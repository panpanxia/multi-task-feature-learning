# Objective : @TODO
# Created by: ece
# Created on: 27.03.2018
""" Routine for decoding the MTFL binary file format """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# Process images of this size. Note that this differs from the original MTFL
# Image size of 40 x 40.
IMAGE_SIZE = 40
# Global constants describing the MTFL data set.
NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 10000
NUM_EXAMPLES_PER_EPOCH_FOR_VAL = 1000
NUM_EXAMPLES_PER_EPOCH_FOR_TEST = 2995


def assign_label_size():
    if FLAGS.task == "5":
        NUM_LABELS = 10
    elif FLAGS.task == "1":
        NUM_LABELS = 1
    elif FLAGS.task == "2":
        NUM_LABELS = 1
    elif FLAGS.task == "3":
        NUM_LABELS = 1
    elif FLAGS.task == "4":
        NUM_LABELS = 5
    return NUM_LABELS


def read_mtfl(filename_queue):
    """
    Reads and parses examples from MTFL data files.


    :param filename_queue: A queue of strings with the filenames to read from.
    :returns: An object representing a single exapmle, with the following fields:
        height: number of rows in the result (40)
        width: numver of columns in the result (40)
        depth: number of color channels in the result (3)
        key: a scalar string Tensor describing the filename & record number
        for this example.
        label: an int32 Tensor with the label in the range 1..5.
        uint8image: a [height, width, depth] uint8 Tensor with the image data.
    """

    class MTFLRecord(object):
        pass

    result = MTFLRecord()
    # Dimensions of the images in the MTFL data set.
    label_bytes = 14  # ---> 4 softmax label 10 landmark label
    result.height = FLAGS.bin_image_size
    result.width = FLAGS.bin_image_size
    result.depth = 1
    image_bytes = result.height * result.width * result.depth
    # Every record consists of a label followed by the image,
    # with a fixed number of bytes for each.
    record_bytes = label_bytes + image_bytes

    # Read a record, getting filenames from the filename queue.
    # No header or footer in the MTFL format, so we leave header bytes.
    # and footer bytes at their defaut 0.
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)
    # Convert from a string to a vector of uint8 that is record bytes long.
    record_bytes = tf.decode_raw(value, tf.uint8)

    # The first bytes represent the label, which we convert from uint8->int32
    # label size: [14]
    result.label = tf.cast(
        tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(
        tf.strided_slice(record_bytes, [label_bytes],
                         [label_bytes + image_bytes]),
        [result.depth, result.height, result.width])

    # Convert from [depth, height, width] to [height, width, depth].
    # image size: [100 100 1]
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])
    return result


def distorted_inputs(filenames, batch_size):
    """
        Construct distorted input for MTFL training using the Reader ops.
        Args:
            data_dir: Path to the MTFL data directory.
            batch_size: Number of images per batch.
        Returns:
            images: Images. 4D Tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
            labels: Labels. 1D Tensor of [batch_size].

    """
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the file names to read.
    filename_queue = tf.train.string_input_producer(filenames)

    with tf.name_scope('data_normalization'):
        # Read examples from files in the filename queue.
        read_input = read_mtfl(filename_queue)
        reshaped_image = tf.cast(read_input.uint8image, tf.float32)

        height = IMAGE_SIZE
        width = IMAGE_SIZE
        # 1.    Set random brightness
        distorted_image = tf.image.random_brightness(reshaped_image,
                                                     max_delta=63)
        # 2.    Set random contrast
        distorted_image = tf.image.random_contrast(distorted_image,
                                                   lower=0.2, upper=1.8)
        # 3.    Subtract off the mean and divide by the variance of the pixels.
        float_image = tf.image.per_image_standardization(distorted_image)
        # Check mean and variance of the images
        # flat_image = tf.image.resize_images(float_image, tf.Variable([40*40, 1]))
        # momentum = tf.nn.moments(flat_image, axes=[0])
        # Set the shapes of tensors.
        float_image = tf.cast(float_image, tf.float32)
        float_image.set_shape([height, width, 1])
        # 4.    Subtract off the mean and divide by variance of the landmarks
        if FLAGS.task == "5":
            landmarks = tf.slice(read_input.label, begin=[4], size=[10])
            # add max and min coordinates to normalize properly
            min_max = tf.Variable([1, width])
            landmarks = tf.concat([landmarks, min_max], 0)
            float_landmarks = tf.cast(landmarks, tf.float32)
            float_landmarks = tf.nn.batch_normalization \
                (float_landmarks,
                 mean=tf.Variable(tf.zeros(tf.shape(float_landmarks))),
                 variance=tf.Variable(tf.ones(tf.shape(float_landmarks))),
                 offset=[0], variance_epsilon=[1e-3],
                 scale=[1 / width])
            float_landmarks = tf.slice(float_landmarks, begin=[0], size=[10])
            read_input.label = float_landmarks
            # Set the shapes of tensors.
            read_input.label.set_shape([10])
        elif FLAGS.task == "1":
            gender_labels = tf.slice(read_input.label, begin=[0], size=[1])
            # Make them 0 and 1 instead of 1 and 2, and DO NOT FORGET TO MAKE TRAINABLE FALSE
            gender_labels = tf.subtract(gender_labels, tf.Variable(1, trainable=False))
            read_input.label = gender_labels
            # Set the shapes of tensors.
            read_input.label.set_shape([1])

        # Ensure that the random shuffling has good mixing parameters:
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                                 min_fraction_of_examples_in_queue)

        print('Filling queue with %d MTFL images before starting to train. '
              'This will take a few minutes.' % min_queue_examples)

    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples, batch_size,
                                           shuffle=True)


# def train_inputs(filenames, batch_size):
#     """
#         Construct distorted input for MTFL training using the Reader ops.
#         Args:
#             data_dir: Path to the MTFL data directory.
#             batch_size: Number of images per batch.
#         Returns:
#             images: Images. 4D Tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
#             labels: Labels. 1D Tensor of [batch_size].
#
#     """
#     # Trim filanames, make last one as validation set
#     filenames = filenames[:-1]
#     # Do exact pre-processes in distroted_inputs but do not use tensorflow
#     # Its itchy!


# def validation_inputs(filenames, batch_size):
#     """
#     Works exact same as the mtfl_input.distorted_inputs
#     :param filenames:
#     :param batch_size:
#     :return:
#     """
#     filenames = [filenames[-1]]
#     # Do exact pre-processes in distroted_inputs but do not use tensorflow
#     #     # Its itchy!


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    """
    Construct a queued batch of images and labels.
    :param image: Tensor 3-D image of [height, width, 3] of type float32.
    :param label: Tensor 1-D type int32
    :param min_queue_examples: int32, minimum number of samples to retrain
    in the queue that provides of batches of examples.
    :param batch_size: Number of images per batch.
    :param shuffle: boolean indicating whether to use a shuffling queue.
    :return:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
    """
    NUM_LABELS = assign_label_size()
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    tf.summary.image('images', images)
    return images, tf.reshape(label_batch, [batch_size, NUM_LABELS])
