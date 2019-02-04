import pickle
import tensorflow as tf

def model_fn(features, labels, mode):
    # tf.keras.layers.InputLayer(input_shape=[32, 32, 3])(features)
    net = tf.keras.layers.Conv2D(16, [3,3], padding="same", use_bias=False, input_shape=(32, 32, 3))(features)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Activation(tf.nn.relu)(net)
    net = tf.keras.layers.MaxPooling2D(2,2)(net)

    net = tf.keras.layers.Conv2D(32, [3,3], padding="same", use_bias=False)(net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Activation(tf.nn.relu)(net)
    net = tf.keras.layers.MaxPooling2D(2,2)(net)

    net = tf.keras.layers.Conv2D(43, [3, 3], padding="same", use_bias=True)(net)
    net = tf.keras.layers.GlobalAveragePooling2D()(net)

    # Compute predictions.

    predicted_classes = tf.argmax(net, 1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(net),
            'logits': net,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=net)

    accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_classes, name='acc_op')

    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)


    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdamOptimizer(0.001)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)