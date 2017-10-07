from datetime import datetime
import tensorflow as tf
import time
from tensorflow.contrib import slim
from tensorflow.python.training.basic_session_run_hooks import CheckpointSaverHook

from nets.vgg import vgg_a
from preprocessing import preprocessing_factory

from datasets import imagenet

batch_size = 64
labels_offset = 1

def create_variable(name, shape, initializer):
  dtype = tf.float32
  return tf.get_variable(name, shape, initializer=initializer, dtype=dtype)


def variable_with_weight_decay(name, shape, stddev):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float32
  var = create_variable(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  return var

NUM_CLASSES = 1000

def inference(images, reuse=False):
    """Build the CIFAR-10 model.

    Args:
      images: Images returned from distorted_inputs() or inputs().

    Returns:
      Logits.
    """

    conv1 = slim.conv2d(images, 3, 64, [3, 3], scope='conv1', reuse=reuse)

    # pool1
    pool1 = slim.max_pool2d(conv1, [2, 2], scope='pool1')

    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')

    # conv2
    conv2 = slim.conv2d(norm1, 64, 128, [3, 3], scope='conv2', reuse=reuse)

    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm2')
    # pool2
    pool2 = slim.max_pool2d(norm2, [2, 2], scope='pool2')

    # conv3
    conv3 = slim.conv2d(pool2, 128, 128, [3, 3], scope='conv3', reuse=reuse)

    # norm3
    norm3 = tf.nn.lrn(conv3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm2')
    # pool2
    pool3 = tf.nn.max_pool(norm3, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    print(pool3.get_shape())

    # local3
    with tf.variable_scope('local3', reuse=reuse) as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(pool3, [batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = variable_with_weight_decay('weights', shape=[dim, 512],
                                             stddev=0.04)
        biases = create_variable('biases', [512], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    # local4
    with tf.variable_scope('local4', reuse=reuse) as scope:
        weights = variable_with_weight_decay('weights', shape=[512, 512],
                                             stddev=0.04)
        biases = create_variable('biases', [512], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)

    # linear layer(WX + b),
    # We don't apply softmax here because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency.
    with tf.variable_scope('softmax_linear', reuse=reuse) as scope:
        weights = variable_with_weight_decay('weights', [512, NUM_CLASSES],
                                             stddev=1 / 192.0)
        biases = create_variable('biases', [NUM_CLASSES],
                                 tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)

    return softmax_linear


class _LoggerHook(tf.train.SessionRunHook):
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


def variable_summaries(name, var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar(name + '/mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar(name + '/stddev', stddev)
        tf.summary.scalar(name + '/max', tf.reduce_max(var))
        tf.summary.scalar(name + '/min', tf.reduce_min(var))
        tf.summary.histogram(name + '/histogram', var)


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
            label -= labels_offset

        #train_image_size = 32
        train_image_size = 224

        preprocessing_name = "vgg"

        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=True)

        image = image_preprocessing_fn(image, train_image_size, train_image_size)

        print(image)
        print(label)

        images_batch, labels_num = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=4,
            capacity=5 * batch_size)

        labels_batch = slim.one_hot_encoding(labels_num, dataset.num_classes - labels_offset)

        images = images_batch #tf.placeholder(tf.float32, shape=(batch_size, train_image_size, train_image_size, 3))
        labels = labels_batch #tf.placeholder(tf.float32, shape=(batch_size, 1000))

        logits, end_points = vgg_a(images)

        logits = tf.clip_by_value(logits, -20, 20)
        #logits = inference(images)

        loss = tf.losses.softmax_cross_entropy(labels, logits, label_smoothing=0.001)

        correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(logits, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tvars = tf.trainable_variables()

        grads = tf.gradients(loss, tvars)

        for var, grad in zip(tvars, grads):
            variable_summaries(var.name, grad)

        train_step = tf.train.RMSPropOptimizer(1e-2, momentum=0.9, decay=0.9, epsilon=0.01).minimize(
            loss, global_step=global_step, var_list=tvars)

        mean_loss = 0
        mean_acc = 0

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('train_logs')

        with tf.train.MonitoredSession(hooks=[CheckpointSaverHook("checkpoints", save_secs=30 * 60)]) as sess:
            for i in range(1000000):
                l, acc, _ = sess.run([loss, accuracy, train_step])

                mean_loss += l / 1000.0
                mean_acc += acc / 1000.0

                if (i + 1) % 1000 == 0:
                    print("i {}, loss: {:.5f} acc: {}".format(i, mean_loss, mean_acc))
                    summary = sess.run(merged)
                    train_writer.add_summary(summary, i)
                    mean_loss = 0
                    mean_acc = 0


if __name__ == '__main__':
    tf.app.run()
