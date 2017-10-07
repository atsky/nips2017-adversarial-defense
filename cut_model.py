import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.python.training.monitored_session import MonitoredSession, MonitoredTrainingSession

from bayesian_inception import wrap_func
from datasets import dataset_factory
from nets import inception_resnet_v2_bayes
from preprocessing import preprocessing_factory

num_classes = 1001

batch_size = 100

num_preprocessing_threads = 4

model_name = "inception_resnet_v2"

is_training = False


def main(_):
    with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()

        ######################
        # Select the dataset #
        ######################
        dataset = dataset_factory.get_dataset(
            "imagenet", "train", "/home/atsky/imagenet-data")

        ####################
        # Select the model #
        ####################
        func = inception_resnet_v2_bayes.inception_resnet_v2_bayes

        network_fn = wrap_func(func)

        # func = inception_v3_bayes

        ##############################################################
        # Create a dataset provider that loads data from the dataset #
        ##############################################################
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            shuffle=False,
            common_queue_capacity=2 * batch_size,
            common_queue_min=batch_size)

        [image, label] = provider.get(['image', 'label'])

        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=False)

        eval_image_size = 299

        image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

        images, labels = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocessing_threads,
            capacity=5 * batch_size)

        ####################
        # Define the model #
        ####################
        logits, _ = network_fn(images)

        variables_to_restore = slim.get_variables_to_restore()

        predictions = tf.argmax(logits, 1)
        labels = tf.squeeze(labels)

        # Define the metrics:
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
            'Recall_5': slim.metrics.streaming_recall_at_k(
                logits, labels, 5),
        })

        # Print the summaries to screen.
        for name, value in names_to_values.items():
            summary_name = 'eval/%s' % name
            op = tf.summary.scalar(summary_name, value, collections=[])
            op = tf.Print(op, [value], summary_name)
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

        num_batches = 100

        checkpoint_path = "./copy/model.ckpt-5281"

        tf.logging.info('Evaluating %s' % checkpoint_path)

        reader = tf.pywrap_tensorflow.NewCheckpointReader(checkpoint_path)

        ops = []

        for var in slim.get_model_variables():
            var_name = var.op.name
            print(var_name)
            assing_op = var.assign(reader.get_tensor(var_name))
            ops.append(assing_op)

        increment_global_step_op = tf.assign(tf_global_step, tf_global_step + 1)

        with MonitoredTrainingSession(checkpoint_dir="./base_checkpoint/") as session:
            session.run(ops)

            for i in range(num_batches):
                session.run(increment_global_step_op)

                a = session.run(list(names_to_updates.values()))
                print(a)


if __name__ == '__main__':
    tf.app.run()
