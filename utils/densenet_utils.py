import collections
import tensorflow as tf

# Learning hyperaparmeters
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
  """A named tuple describing a DenseNet block.

  Its parts are:
    scope: The scope of the `Block`.
    unit_fn: The DenseNet unit function which takes as input a `Tensor` and
      returns another `Tensor` with the output of the DenseNet unit.
    args: A list of length equal to the number of units in the `Block`. The list
      contains one (depth, depth_bottleneck, stride) tuple for each unit in the
      block to serve as argument to unit_fn.
  """

@tf.contrib.framework.add_arg_scope
def stack_blocks_dense(net,
                       growth_rate,
                       is_training=True,
                       bottleneck_number_feature_maps=None,
                       dropout_keep_prob=0.2,
                       **kwargs):
  """
  Stacks DenseNet units
  bottleneck: BN-ReLu-Conv(1x1)
  BN-ReLu-Conv(3x3)
  """
  # bottleneck if defined
  if bottleneck_number_feature_maps:
    with tf.variable_scope('batch_norm_0'):
      net = tf.layers.batch_normalization(
        inputs=net,
        axis=-1,
        fused=True,
        center=True,
        scale=True,
        training=is_training,
        momentum=_BATCH_NORM_DECAY,
        epsilon=_BATCH_NORM_EPSILON,
      )
      net = tf.nn.relu(net)
    with tf.variable_scope('bottleneck'):
      net = tf.layers.conv2d(net,
                             bottleneck_number_feature_maps,
                             kernel_size=1,
                             padding='same',
                             use_bias=False,
                             kernel_regularizer=None,
                             activation=tf.identity)
    if dropout_keep_prob:
      with tf.variable_scope("dropout_0"):
        net = tf.layers.dropout(net, rate=dropout_keep_prob, training=is_training,)

  # convolution
  with tf.variable_scope('batch_norm_1'):
    net = tf.layers.batch_normalization(net, training=is_training)
    net = tf.nn.relu(net)
  net = tf.layers.conv2d(net, growth_rate,
                         use_bias=False,
                         padding='same',
                         kernel_size=3,
                         kernel_regularizer=None,
                         activation=tf.identity)
  if dropout_keep_prob:
    with tf.variable_scope("dropout"):
      net = tf.layers.dropout(net, rate=dropout_keep_prob, training=is_training,)
  return net


@tf.contrib.framework.add_arg_scope
def add_transition_down_layer(net,
                              is_training,
                              dropout_keep_prob=0.2,
                              compression_factor=1.0):
  """
  Transition Down
  BN-Conv(1x1) of same number of filters-Avg Pooling 2x2
  """
  def _int_shape(layer):
    return layer.get_shape().as_list()

  with tf.variable_scope('transition_layer_down'):
    depth_in = _int_shape(net)[-1]
    net = tf.layers.batch_normalization(
        inputs=net,
        axis=-1,
        fused=True,
        center=True,
        scale=True,
        training=is_training,
        momentum=_BATCH_NORM_DECAY,
        epsilon=_BATCH_NORM_EPSILON,
    )
    net = tf.layers.conv2d(net, depth_in,
                           kernel_size=1,
                           padding='same',
                           use_bias=False,
                           kernel_regularizer=None,
                           activation=tf.identity)
  if dropout_keep_prob:
    net = tf.layers.dropout(net, rate=dropout_keep_prob, training=is_training)
  with tf.variable_scope('pool'):
    net = tf.layers.average_pooling2d(net, pool_size=2, strides=2, padding="same")
  return net


@tf.contrib.framework.add_arg_scope
def add_transition_up_layer(net,
                            stride=2,
                            kernel_size=3,
                            activation=None,
                            scope=None):
  # no activation between layers in upsampling
  def _int_shape(layer):
    return layer.get_shape().as_list()
  depth_in = _int_shape(net)[-1]
  return tf.layers.conv2d_transpose(net,
                                    depth_in,
                                    kernel_size=kernel_size,
                                    use_bias=False,
                                    strides=stride,
                                    padding='same',
                                    activation=activation)




