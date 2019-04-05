from collections import OrderedDict
from pathlib import Path

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from .data import create_mnist_dataset
from .model import mnist_model

# for reproducibility
SEED = 42


def model_fn(features, labels, mode, params):
    """
    Model function for the Estimator. See tf.estimator.Estimator documentation for details.

    :param features: The first param of the input_fn output
    :param labels: The second param of the input_fn output
    :param mode: one of tf.estimator.ModeKeys
    """

    # see if we're in training mode
    training = (mode == tf.estimator.ModeKeys.TRAIN)

    # create the model and get the logits output tensor
    with tf.device('gpu'):
        logits = mnist_model(features, training)
        logits = tf.identity(logits, name='logits')

        # build loss (softmax cross entropy)
        loss = tf.losses.softmax_cross_entropy(labels, logits, scope='loss')

        # Build the train operation using the Adam optimizer. Optionally allows a 'learning_rate' hyper-param
        # We'll also make sure the global 'global_step' variable is updated on train.
        train_op = tf.train.AdamOptimizer(learning_rate=params.get('learning_rate', 0.001)) \
            .minimize(loss, global_step=tf.global_variables('global_step')[0])

    # add a saver to enable saving all variables defined so far
    saver = tf.train.Saver()
    scaffold = tf.train.Scaffold(saver=saver)

    # use accuracy for evaluation
    predictions = tf.nn.softmax(logits, axis=1)
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(labels, predictions, name='accuracy')
    }

    # return all these tensors as a EstimatorSpec object
    return tf.estimator.EstimatorSpec(mode, predictions, loss, train_op, eval_metric_ops, scaffold=scaffold)


def do_train(work_dir: Path, epochs: int, batch_size=2, **kwargs):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    # sess = tf.InteractiveSession()
    sess.__enter__()
    work_dir.mkdir(exist_ok=True, parents=True)
    global assignments
    assignments = tf.contrib.lookup.MutableHashTable(key_dtype=tf.int32, value_dtype=tf.int32, default_value=-1)
    # assignments = {}
    keys = []
    values = []
    for i in range(600):
        if i <= 300:
            # print(type(tf.constant(0, dtype=tf.int32)))
            # assignments[i] = tf.constant(0, dtype=tf.int32)
            # key = tf.constant(i, dtype=tf.int32)
            # value = tf.constant(0, dtype=tf.int32)
            # sess.run(assignments.insert(key, value))
            keys.append(tf.Variable(i, dtype=tf.int32))
            values.append(tf.Variable(0, dtype=tf.int32))
        else:
            # assignments[i] = tf.constant(1, dtype=tf.int32)
            # key = tf.constant(i, dtype=tf.int32)
            # value = tf.constant(1, dtype=tf.int32)
            # sess.run(assignments.insert(key, value))
            keys.append(tf.Variable(i, dtype=tf.int32))
            values.append(tf.Variable(1, dtype=tf.int32))
    sess.run(tf.global_variables_initializer())
    sess.run(assignments.insert(keys, values))
    # assignments = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(keys, vals),-1)
    # assignments.init.run()
    # create datasets for training and evaluation
    train_ds, train_size = create_mnist_dataset(batch_size, 'train', sess, assignments)
    print("finish train")
    eval_ds, eval_size = create_mnist_dataset(batch_size, 'val', sess, assignments)
    print("finish eval")
    # create a reinitializable with the correct structure (but unbounded to any dataset)
    # iterator: tf.data.Iterator = tf.data.Iterator.from_structure(train_ds.output_types, train_ds.output_shapes)
    iterator = train_ds.make_initializable_iterator()
    next_batch = iterator.get_next()
    img_ids, images = next_batch[0]
    labels = next_batch[1]
    # batch_assignments = next_batch[1]
    # batch_assignments = next_batch[1]

    print(img_ids.shape)
    print(images.shape)
    print(labels.shape)

    # initialize global control variables
    global_step = tf.Variable(0, name='global_step', dtype=tf.int64, trainable=False)
    tf.add_to_collection(tf.GraphKeys.GLOBAL_STEP, global_step)

    # make sure NN ops run in GPU
        
    # create a training placeholder to control conditional graph execution of layers such as
    # dropout or batch_norm
    training = tf.placeholder(tf.bool, shape=())

    # create the model and get the logits and prediction tensors
    logits = mnist_model(images, training=training)
    predictions = tf.nn.softmax(logits, name='predictions')

    # build loss (softmax cross entropy)
    loss = tf.losses.softmax_cross_entropy(labels, logits, scope='loss')

    # Build the train operation using the Adam optimizer.
    # We'll also make sure the global 'global_step' variable is updated on train.
    train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss, global_step=global_step)

    # Setup metrics
    accuracy, accuracy_op = tf.metrics.accuracy(tf.argmax(predictions, axis=1), tf.argmax(labels, axis=1))
    mean_loss, mean_loss_op = tf.metrics.mean(loss)

    # get update operations (required for BatchNorm layers)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    update_ops += [accuracy_op, mean_loss_op]

    # merge update_ops and train_op into a single operation
    train_op = tf.group(update_ops + [train_op])

    random = np.random.RandomState(SEED)

    sess.run(tf.global_variables_initializer())

    for epoch in range(1, epochs + 1):

        # initialize iterator for train dataset. Shuffle dataset first
        # ds = train_ds.shuffle(128, seed=random.randint(0, 1024))
        ds = train_ds
        sess.run(iterator.make_initializer(ds))
        # zero-out metrics, they're kept in the local variables collection
        sess.run(tf.local_variables_initializer())
        

        # start a progress bar and iterate until dataset exhaustion
        tqdm.write(f'Training on epoch {epoch}')
        bar = tqdm(total=train_size)
        print("148")
        count = 0
        while True:
            if count == 10:
                for i in range(600):
                    if i <= 300:
                        # print("lookup key: ", sess.run(assignments.lookup(keys[i])))
                        # key = tf.constant(i, dtype=tf.int32)
                        # value = tf.Variable(1, dtype=tf.int32)
                        # sess.run(assignments.insert(keys[i], value))
                        sess.run(tf.assign(values[i], 1))
                    else:
                        # key = tf.constant(i, dtype=tf.int32)
                        # value = tf.Variable(0, dtype=tf.int32)
                        # sess.run(assignments.insert(keys[i], value))
                        sess.run(tf.assign(values[i], 0))
                # for i in range(600):
                #     if i <= 300:
                #         sess.run(assignments[i].assign(1))
                #     else:
                #         sess.run(assignments[i].assign(0))
                sess.run(assignments.insert(keys, values))
                sess.run(iterator.make_initializer(ds))
            # if count == 20:
            #     for i in range(600):
            #         if i <= 300:
            #             sess.run(assignments[i].assign(0))
            #         else:
            #             sess.run(assignments[i].assign(1))
                # sess.run(iterator.make_initializer(ds))
            if count >= 30:
                raise
            try:
                print("count: ", count)
                # print("assignments.size(): ", assignments.size().eval())
                print("assignments.lookup(keys[146]): ", sess.run(assignments.lookup(keys[146])))
                print("assignments.lookup(keys[400]): ", sess.run(assignments.lookup(keys[400])))
                # raise
                # run train iteration
                # print("166")
                _, preds_v = sess.run([train_op, predictions], feed_dict={training: True})
                print(sess.run(img_ids))
                # print(sess.run(batch_assignments))
                # update progress bar with batch size
                batch_size = preds_v.shape[0]
                bar.update(batch_size)

                # update progress bar postfix info with metrics
                accuracy_v, mean_loss_v = sess.run([accuracy, mean_loss])
                postfix = OrderedDict(loss=f'{mean_loss_v:.4f}', accuracy=f'{accuracy_v:.4f}')
                bar.set_postfix(postfix)

            except (tf.errors.OutOfRangeError, StopIteration):
                print("179")
                bar.close()
                break
            count += 1

        # initialize iterator for validation dataset. No need to shuffle
        sess.run(iterator.make_initializer(eval_ds))

        # zero-out metrics for evaluation
        sess.run(tf.local_variables_initializer())

        tqdm.write(f'Evaluating on epoch {epoch}')
        bar = tqdm(total=eval_size)
        while True:
            try:
                # run eval to update metrics
                accuracy_v, mean_loss_v, preds_v = sess.run(
                    [accuracy_op, mean_loss_op, predictions], feed_dict={training: False})

                # update progress bar with batch size
                batch_size = preds_v.shape[0]
                bar.update(batch_size)

                # update progress bar postfix info with metrics
                postfix = OrderedDict(loss=f'{mean_loss_v:.4f}', accuracy=f'{accuracy_v:.4f}')
                bar.set_postfix(postfix)

            except (tf.errors.OutOfRangeError, StopIteration):
                bar.close()
                break

        tqdm.write('------')