import tensorflow as tf 


class DenseNet(object):
    def __init__(self, nb_blocks, filters, dropout=0.2):
        self.nb_blocks = nb_blocks
        self.filters = filters
        self.dropout = dropout

    def bottleneck_layer(self, net, scope):
        with tf.name_scope(scope):
            net = tf.keras.layers.BatchNormalization()(net)
            net = tf.keras.layers.Activation(tf.nn.relu)(net)
            net = tf.keras.layers.Conv2D(filters=4*self.filters, kernel_size=1, padding='same', use_bias=False)(net)
            net = tf.keras.layers.Dropout(self.dropout)(net)

            net = tf.keras.layers.BatchNormalization()(net)
            net = tf.keras.layers.Activation(tf.nn.relu)(net)

            net = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=3, padding='same', use_bias=False)(net)
            net = tf.keras.layers.Dropout(self.dropout)(net)

            return net

    def transition_layer(self, net, scope):
        with tf.name_scope(scope):
            net = tf.keras.layers.BatchNormalization()(net)
            net = tf.keras.layers.Activation(tf.nn.relu)(net)
            net = tf.keras.layers.Conv2D(self.filters, 1)(net)
            net = tf.keras.layers.Dropout(self.dropout)(net)
            net = tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=2, padding='valid')(net)

            return net

    def dense_block(self, net, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            layers_concat = [net]
            net = self.bottleneck_layer(net, scope= layer_name + '_bottleN_' + '0')

            layers_concat.append(net)

            for i in range(nb_layers - 1):
                # net = tf.concat(-1, layers_concat)
                net = tf.keras.layers.Concatenate()(layers_concat)
                net = self.bottleneck_layer(net, scope=layer_name + '_bottleN_' + str(i + 1))
                layers_concat.append(net)

            # net = tf.concat(-1, layers_concat)
            net = tf.keras.layers.Concatenate()(layers_concat)

            return net

    def __call__(self, net):
        net = tf.keras.layers.Convolution2D(filters = 2*self.filters, kernel_size=7, strides=2, use_bias=False)(net)

        net = self.dense_block(net=net, nb_layers=6, layer_name='dense_1')
        net = self.transition_layer(net, scope='trans1')

        net = self.dense_block(net=net, nb_layers=12, layer_name='dense_2')
        net = self.transition_layer(net, scope='trans_2')

        net = self.dense_block(net=net, nb_layers=48, layer_name='dense_3')
        net = self.transition_layer(net, scope='trans_3')

        net = self.dense_block(net=net, nb_layers=32, layer_name='dense_final')

        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Activation(tf.nn.relu)(net)
        net = tf.keras.layers.GlobalAvgPool2D()(net)
        net = tf.keras.layers.Flatten()(net)
        net = tf.keras.layers.Dense(43)(net)

        return net

        
def model_fn(features, labels, mode):

    # TODO: Change to Constants
    dense_net = DenseNet(2, 24)
    
    net = tf.keras.layers.InputLayer(input_shape=[32, 32, 3])(features)
    net = dense_net(net)

    predicted_classes = tf.argmax(net, 1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes,
            'probabilities': tf.nn.softmax(net),
            #'logits': net,
            'features': features
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=net)

    accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_classes, name='acc_op')

    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])


    if mode == tf.estimator.ModeKeys.EVAL:
        # Create a SummarySaverHook
        # eval_summary_hook = tf.train.SummarySaverHook(
        #                                 save_steps=1,
        #                                 output_dir= "models/data_norm" + "/eval_core",
        #                                 summary_op=tf.summary.merge_all())

        # return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics, evaluation_hooks=[eval_summary_hook])
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics, )


    assert mode == tf.estimator.ModeKeys.TRAIN


    # tf.summary.image("images", features)

    optimizer = tf.train.AdamOptimizer(0.0001)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
