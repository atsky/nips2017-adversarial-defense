"""Implementation of sample defense.

This defense loads inception v3 checkpoint and classifies all images
using loaded checkpoint.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import time
from scipy.misc import imread

import tensorflow as tf

from nets import inception_resnet_v2_bayes
import inception_resnet_v2

slim = tf.contrib.slim

tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path', './base_checkpoint/', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_file', '', 'Output file to save labels.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 128, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS


def load_data(input_dir):
    data = []
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
        with tf.gfile.Open(filepath, 'rb') as f:
            image = imread(f, mode='RGB').astype(np.float) / 255.0

        data.append((os.path.basename(filepath), image * 2.0 - 1.0))
    return data


def load_images(data, batch_shape):
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]

    data_index = 0
    while True:
        filename, image = data[data_index]
        data_index += 1
        if data_index >= len(data):
            data_index = 0

        filenames.append(filename)
        images[idx, :, :, :] = image

        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0


def main(_):
    start_time = time.time()

    data = load_data(FLAGS.input_dir)

    data_size = len(data)
    print("data size: {}".format(data_size))

    batch_size = np.minimum(FLAGS.batch_size, data_size)
    batch_shape = [batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    num_classes = 1001

    tf.logging.set_verbosity(tf.logging.INFO)

    probs_dict = {}

    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)

        func = inception_resnet_v2_bayes.inception_resnet_v2_bayes

        with slim.arg_scope(inception_resnet_v2_bayes.inception_resnet_v2_arg_scope_bayes()):
            logits, end_points = func(x_input, num_classes=num_classes, is_training=False)

        probs = tf.nn.softmax(logits)

        # Run computation
        saver = tf.train.Saver(slim.get_model_variables())
        session_creator = tf.train.ChiefSessionCreator(
            scaffold=tf.train.Scaffold(saver=saver),
            checkpoint_dir="./model1/",
            master=FLAGS.master)

        total_images = 0

        with tf.train.MonitoredSession(session_creator=session_creator) as sess:
            for filenames, images in load_images(data, batch_shape):
                labels_probs = sess.run(probs, feed_dict={x_input: images})
                for filename, p in zip(filenames, labels_probs):
                    if filename in probs_dict:
                        probs_dict[filename] = probs_dict[filename] + np.array(p) * 0.9
                    else:
                        probs_dict[filename] = np.array(p) * 0.9

                #print("Time: {} {}".format(time.time() - start_time, total_images))

                total_images += batch_size

                if total_images > data_size * 128:
                    break

                if time.time() - start_time > 200:
                    break

    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)

        func = inception_resnet_v2.inception_resnet_v2

        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits, end_points = func(x_input, num_classes=num_classes, is_training=False)

        probs = tf.nn.softmax(logits)

        # Run computation
        saver = tf.train.Saver(slim.get_model_variables())
        session_creator = tf.train.ChiefSessionCreator(
            scaffold=tf.train.Scaffold(saver=saver),
            checkpoint_dir="./model2/",
            master=FLAGS.master)

        total_images = 0

        with tf.train.MonitoredSession(session_creator=session_creator) as sess:
            for filenames, images in load_images(data, batch_shape):
                labels_probs = sess.run(probs, feed_dict={x_input: images})
                for filename, p in zip(filenames, labels_probs):
                    if filename in probs_dict:
                        probs_dict[filename] = probs_dict[filename] + p
                    else:
                        probs_dict[filename] = np.array(p)

                #print("Time: {} {}".format(time.time() - start_time, total_images))

                total_images += batch_size

                if total_images > data_size * 128:
                    break

                if time.time() - start_time > 450:
                        break

    with tf.gfile.Open(FLAGS.output_file, 'w') as out_file:
        for filename, _ in data:
            label = np.argmax(probs_dict[filename])
            out_file.write('{0},{1}\n'.format(filename, label))


if __name__ == '__main__':
    tf.app.run()
