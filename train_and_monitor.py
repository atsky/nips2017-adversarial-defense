from datetime import datetime

import numpy
import tensorflow as tf
import time
from tensorflow.contrib import slim
from tensorflow.python.training.basic_session_run_hooks import CheckpointSaverHook

from bayesian_inception import wrap_func

from nets import inception_resnet_v2_bayes

from preprocessing import preprocessing_factory

from datasets import imagenet

batch_size = 16

class LoadHook(tf.train.SessionRunHook):
    """Logs loss and runtime."""

    def __init__(self, init_fn):
        self.init_fn = init_fn

    def after_create_session(self, session, coord):
        self.init_fn(session)


class LoggerHook(tf.train.SessionRunHook):
    """Logs loss and runtime."""

    def __init__(self, loss):
        self.loss = loss

    def begin(self):
        self._step = -1
        self._start_time = time.time()

    def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(self.loss)  # Asks for loss value.

    def after_run(self, run_context, run_values):
        if self._step % 10 == 0:
            current_time = time.time()
            duration = current_time - self._start_time
            self._start_time = current_time

            loss_value = run_values.results
            examples_per_sec = 0  # FLAGS.log_frequency * FLAGS.batch_size / duration
            sec_per_batch = 0  # float(duration / FLAGS.log_frequency)

            format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                          'sec/batch)')
            print(format_str % (datetime.now(), self._step, loss_value,
                                examples_per_sec, sec_per_batch))

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
        global_step = tf.contrib.framework.get_or_create_global_step()

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

        preprocessing_name = "inception_v3"

        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=True)

        image = image_preprocessing_fn(image, train_image_size, train_image_size)

        images_batch, labels_num = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=4,
            capacity=5 * batch_size)

        labels_batch = slim.one_hot_encoding(labels_num, dataset.num_classes)

        images = images_batch
        labels = labels_batch

        func = inception_resnet_v2_bayes.inception_resnet_v2_bayes

        network_fn = wrap_func(func)

        logits, end_points = network_fn(images)

        if 'AuxLogits' in end_points:
            slim.losses.softmax_cross_entropy(
                end_points['AuxLogits'], labels,
                label_smoothing=0.01, weights=0.1,
                scope='aux_loss')

        slim.losses.softmax_cross_entropy(
            logits, labels, label_smoothing=0.01, weights=1.0)

        clone_losses = tf.get_collection(tf.GraphKeys.LOSSES)

        clone_loss = tf.add_n(clone_losses, name='loss')

        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        regularization_loss = tf.add_n(regularization_losses,
                                       name='regularization_loss')

        loss = clone_loss + regularization_loss

        tf.summary.scalar('loss', clone_loss)

        tf.summary.scalar('regularization_loss', regularization_loss)

        correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(logits, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        ema = tf.train.ExponentialMovingAverage(decay=0.99)

        maintain_averages_op = ema.apply([clone_loss, accuracy])

        ema_loss = ema.average(clone_loss)
        ema_accuracy = ema.average(accuracy)

        tf.summary.scalar('ema_loss', ema_loss)
        tf.summary.scalar('ema_accuracy', ema_accuracy)

        tvars = tf.trainable_variables()

        grads = tf.gradients(loss, tvars)

        #g_stats = [tf.reduce_mean(tf.square(g)) for g in grads]

        tvars_names = [t.name for t in tvars]

        train_step = tf.train.RMSPropOptimizer(1e-4, momentum=0.9, decay=0.9, epsilon=1).minimize(
            loss, global_step=global_step, var_list=tvars)

        # Add summaries for variables.
        for variable in slim.get_model_variables():
            if "sd" in variable.op.name:
                tf.summary.histogram(variable.op.name, variable)

        for name, gr in zip(tvars_names, grads):
            if "sd" in name:
                tf.summary.histogram("grad/" + name, gr)


        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('train_logs')

        init_fn = get_init_fn()

        saverHook = CheckpointSaverHook(checkpoint_dir="pretrain", save_secs=10 * 60)

        with tf.train.MonitoredTrainingSession(
                hooks=[LoadHook(init_fn), saverHook]) as sess:

            for i in range(10000):
                summary = sess.run(merged)
                train_writer.add_summary(summary, i)

                #print("global_norm: {}".format(sess.run(global_norm)))
                #g_vals = sess.run(g_stats)

                #for n, g in zip(tvars_names, g_vals):
                #    print(n, g)

                gs, l, rl, acc, _, _ = sess.run([global_step, clone_loss, regularization_loss, accuracy, train_step, maintain_averages_op])

                print("i {}, loss: {:.5f} reg: {:.5f}, acc: {}".format(gs, l, rl, acc))


if __name__ == '__main__':
    tf.app.run()
