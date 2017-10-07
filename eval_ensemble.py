
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from bayesian_inception import wrap_func
from nets import inception_resnet_v2_bayes
from preprocessing import preprocessing_factory

from datasets import imagenet

batch_size = 64

class LoadHook(tf.train.SessionRunHook):
    """Logs loss and runtime."""

    def __init__(self, init_fn):
        self.init_fn = init_fn

    def after_create_session(self, session, coord):
        self.init_fn(session)


def get_init_fn():
    """Returns a function run by the chief worker to warm-start the training.

    Note that the init_fn is only run when initializing the model during the very
    first global step.

    Returns:
      An init function run by the supervisor.
    """

    variables_to_restore = []
    for var in slim.get_model_variables():
        variables_to_restore.append(var)

    checkpoint_path = tf.train.latest_checkpoint("./base_checkpoint")

    tf.logging.info('Fine-tuning from %s' % checkpoint_path)

    return slim.assign_from_checkpoint_fn(
        checkpoint_path,
        variables_to_restore,
        ignore_missing_vars=False)


def main(_):
    with tf.Graph().as_default():
        dataset = imagenet.get_split(
            'train',
            '/home/atsky/imagenet-data')

        with tf.device('/cpu:0'):
            provider = slim.dataset_data_provider.DatasetDataProvider(
                dataset,
                shuffle=True,
                common_queue_capacity=2 * batch_size,
                common_queue_min=batch_size)

            [image, label] = provider.get(['image', 'label'])

        train_image_size = 299

        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            "inception_resnet_v2",
            is_training=False)

        image = image_preprocessing_fn(image, train_image_size, train_image_size)

        images_batch, labels_num = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=4,
            capacity=5 * batch_size)

        labels_batch = labels_num

        func = inception_resnet_v2_bayes.inception_resnet_v2_bayes

        network_fn = wrap_func(func)

        batch_shape = [batch_size, train_image_size, train_image_size, 3]

        x_input = tf.placeholder(tf.float32, shape=batch_shape)

        logits, end_points = network_fn(x_input)

        probs = tf.nn.softmax(logits)

        init_fn = get_init_fn()

        total_accuracy = []

        with tf.train.MonitoredTrainingSession(
                hooks=[LoadHook(init_fn)]) as sess:

            for i in range(10000):
                images, labels = sess.run([images_batch, labels_batch])

                sum_probs = None

                for k in range(20):
                    labels_probs = sess.run(probs, feed_dict={x_input: images})

                    if sum_probs is None:
                        sum_probs = labels_probs
                    else:
                        sum_probs += labels_probs

                acc = 0.0

                for p, l in zip(sum_probs, labels):
                    if np.argmax(p) == l:
                        acc += 1

                acc /= len(labels)
                print("accuracy: {}".format(acc))

                total_accuracy.append(acc)

                std = np.std(total_accuracy) / np.sqrt(len(total_accuracy))
                print("mean accuracy: {} interval: {} {}".format(
                    np.mean(total_accuracy),
                    np.mean(total_accuracy) - std * 2,
                    np.mean(total_accuracy) + std * 2))




if __name__ == '__main__':
    tf.app.run()
