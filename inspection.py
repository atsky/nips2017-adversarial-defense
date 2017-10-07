"""Implementation of sample defense.

This defense loads inception v3 checkpoint and classifies all images
using loaded checkpoint.
"""

import tensorflow as tf
from tensorflow.contrib.slim.nets import inception

slim = tf.contrib.slim


def main(_):
  batch_shape = [16, 299, 299, 3]
  num_classes = 1001

  tf.logging.set_verbosity(tf.logging.INFO)

  with tf.Graph().as_default():
    # Prepare graph
    x_input = tf.placeholder(tf.float32, shape=batch_shape)

    with slim.arg_scope(inception.inception_v3_arg_scope()):
      _, end_points = inception.inception_v3(
          x_input, num_classes=num_classes, is_training=False)

    predicted_labels = tf.argmax(end_points['Predictions'], 1)

    # Run computation
    saver = tf.train.Saver(slim.get_model_variables())
    session_creator = tf.train.ChiefSessionCreator(
        scaffold=tf.train.Scaffold(saver=saver),
        checkpoint_filename_with_path="inception_v3.ckpt",
        master="")

    with tf.train.MonitoredSession(session_creator=session_creator) as sess:
        optim_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        for v in optim_vars:
            print(v.shape)

        #labels = sess.run(predicted_labels, feed_dict={x_input: images})


if __name__ == '__main__':
  tf.app.run()
