from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import os
import tensorflow as tf

import mtfl

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', os.path.join(os.getcwd(), 'mtfl_train_log'),
                           """ Directory where to write event logs and checkpoint""")
tf.app.flags.DEFINE_integer("max_steps", 100000,
                            """ Number of batches to run""")
tf.app.flags.DEFINE_boolean("log_device_placement", False,
                            """ Whether to log device placement""")
tf.app.flags.DEFINE_integer("log_frequency", 5,
                            """ How often to log results to the console""")
tf.app.flags.DEFINE_integer("validation_frequency", 20,
                            """ How often do you want to evaluate validation set""")
tf.app.flags.DEFINE_string("data_set", "MTFL",
                           """ Which data set to use""")
tf.app.flags.DEFINE_string("bin_path", os.path.join(os.getcwd(), 'data/MTFL-batches-bin'),
                           """ Where to create bins, 
                           will not be used if bins are already exists""")
tf.app.flags.DEFINE_string("log_dir", os.path.join(os.getcwd(), "logs"),
                           """ Where to put non train related logs""")
tf.app.flags.DEFINE_integer("bin_image_size", 40,
                            """ Image size to put in bins remember it is raw""")
tf.app.flags.DEFINE_string("task", "1", """ 1 for gender classification, 2 for smile, 
                            3 for glasses, 4 for head pose, 5 for landmarks """)


# Think of train just as main:
# Sessiona ne verirsen onu bekleyebilirsin
def train():
    """ Train MTFL for a number of steps."""

    # Get images and labels for MTFL
    # image size: 128x40x40x1
    # label size: 128x10 veya 128x1
    train_images, train_labels = mtfl.train_inputs()
    val_images, val_labels = mtfl.validation_inputs()

    with tf.Graph().as_default():
        # Create place holder for that you will feed in to the session
        images = tf.placeholder(tf.float32, shape=(128, 40, 40, 1))
        labels = tf.placeholder(tf.int32, shape=(128, 1))

        global_step = tf.train.get_or_create_global_step()
        # Build a graph that computes the logits predictions from the
        # inference model.
        logits = mtfl.inference(images)
        # Calculate loss.
        loss = mtfl.classification_loss(logits, labels)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = mtfl.train(loss, global_step)

        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime."""

            def begin(self):
                self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(loss)  # Asks for loss value.

            def after_run(self, run_context, run_values):
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), self._step, loss_value,
                                        examples_per_sec, sec_per_batch))

        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=FLAGS.train_dir,
                hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                       tf.train.NanTensorHook(loss),
                       _LoggerHook()],
                config=tf.ConfigProto(
                    log_device_placement=FLAGS.log_device_placement)) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op, {images: train_images, labels: train_labels})
                # Open for Debug
                # coord = tf.train.Coordinator()
                # threads = tf.train.start_queue_runners(sess=mon_sess, coord=coord)
                # print(mon_sess.run(eben))
                # print(eben.dtype)


def main(argv=None):
    # Make bins outta data
    mtfl.maybe_make_bins()
    # Create check out folders for train
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


def debug():
    """
    This code block shows what is inside of a tensor.
    This one is golden :)
    """
    with tf.Graph().as_default():
        images, labels = mtfl.distorted_inputs()
        image = images[0]
        mean, variance = tf.nn.moments(image, axes=[0])

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            print(image.shape)
            print(sess.run([mean, variance]))


if __name__ == '__main__':
    tf.app.run()
