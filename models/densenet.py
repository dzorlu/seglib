import tensorflow as tf

from utils import densenet_utils
import collections
from utils import utils

class Encoder(
    collections.namedtuple("Encoder", ("graph", "output"))):
  pass


class BaseModel(object):
  def __init__(self,
               growth_rate,
               is_training=True,
               global_pool=False,
               reuse=None,
               num_classes=None,
               spatial_squeeze=True):
    self.growth_rate = growth_rate
    self.reuse = reuse
    self.is_training = is_training
    self.global_pool = global_pool
    self.num_classes = num_classes
    self.spatial_squeeze = spatial_squeeze

  def encode(self,
            image,
            scope,
            num_units,
            bottleneck_number_feature_maps=None,
            dropout_keep_prob=0.2):
    """
    Args:
    	inputs: image.
      scope: The scope of the block.
      bottleneck_number_feature_maps: If not None, 
        1x1 bottleneck reduces the output
      dropout_rate: dropout rate to apply for each conv unit in training.
      num_units: Array of number of units in each block.
    """
    graph = tf.Graph()
    rate = self.growth_rate
    bottleneck_maps = bottleneck_number_feature_maps
    inputs = image
    with tf.variable_scope(scope, 'densenet', [inputs], reuse=self.reuse) as sc:
      net = inputs
      initial_nb_layers = self.growth_rate * 2
      with tf.variable_scope("conv1"):
        net = tf.layers.conv2d(net,
                               initial_nb_layers,
                               use_bias=False,
                               kernel_size=7,
                               padding='same',
                               strides=2)
      with tf.variable_scope("pool1"):
        net = tf.layers.max_pooling2d(net, [3, 3], strides=2, padding='same')
      # consecutive denseblocks each followed by a transition layer.
      for bn, num_units_in_block in enumerate(num_units):
        with tf.variable_scope("block_{}".format(bn + 1), values=[net]):
          for un, unit in enumerate(range(num_units_in_block)):
            with tf.variable_scope("unit_{}".format(un + 1), values=[net]):
              output = densenet_utils.stack_blocks_dense(net,
                                                         is_training=self.is_training,
                                                         growth_rate=rate,
                                                         bottleneck_number_feature_maps=bottleneck_maps,
                                                         dropout_keep_prob=dropout_keep_prob)

              net = tf.concat(axis=3, values=[net, output])
          # the last layer does not have a transition layer
          if bn + 1 != len(num_units):
            # the output of the dense block. add to the collection for upsampling
            # exclude the bottleneck (the last layer)
            tf.add_to_collection('skip_connections', net)
            # downsample
            net = densenet_utils.add_transition_down_layer(net, is_training=self.is_training)
      # if self.global_pool:
      #   # Global average pooling.
      #   net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
      # if self.num_classes is not None:
      #   with tf.variable_scope('logits'):
      #     net = tf.layers.conv2d(net,
      #                            self.num_classes,
      #                            kernel_size=1,
      #                            kernel_regularizer=None,
      #                            activation=tf.identity)
      #   if self.spatial_squeeze:
      #     net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
      # # Convert end_points_collection into a dictionary of end_points.
      # if self.num_classes is not None:
      #   with tf.variable_scope('softmax'):
      #     net = tf.nn.softmax(net)
      return Encoder(graph=net.graph, output=net)

  def decode(self,
             encoder,
             rate,
             num_units,
             dropout_keep_prob=0.2):
    graph = encoder.graph
    with graph.as_default():
      net = encoder.output
      skip_connections = encoder.graph.get_collection('skip_connections')
      if len(num_units) != len(skip_connections):
        raise ValueError("num units must match skip connections")
      # start from the last layer
      for bn, (num_units_in_block, skip_connection) in enumerate(zip(num_units, reversed(skip_connections))):
        with tf.variable_scope("block_{}".format(bn + 1), values=[net]):
          # transition up the bottleneck layer
          net = densenet_utils.add_transition_up_layer(net)
          # concat with next skip connection.
          net = tf.concat(axis=3, values=[net, skip_connection])
          # denseblock
          for un, unit in enumerate(range(num_units_in_block)):
            with tf.variable_scope("unit_{}".format(un + 1), values=[net]):
              output = densenet_utils.stack_blocks_dense(net,
                                                         growth_rate=rate,
                                                         dropout_keep_prob=dropout_keep_prob)
              # unlike downsampling path, do not concatenate the input layer within the dense block
              if un > 0:
                net = tf.concat(axis=3, values=[net, output])
              else:
                net = output
        with tf.variable_scope('logits'):
          net = tf.layers.conv2d(net,
                                 self.num_classes,
                                 kernel_size=1,
                                 activation=tf.identity)
      return net


