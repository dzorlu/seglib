import tensorflow as tf

from utils import densenet_utils
import collections
from utils import utils

class Encoder(
    collections.namedtuple("Encoder", ("graph", "output"))):
  pass

class Decoder(
    collections.namedtuple("Decoder", ("graph", "output"))):
  pass


class DenseNet(object):
  def __init__(self,
               growth_rate,
               dropout_keep_prob,
               number_classes,
               is_training=True,
               reuse=None,):
    self.growth_rate = growth_rate
    self.reuse = reuse
    self.is_training = is_training
    self.dropout_keep_prob = dropout_keep_prob
    self.number_classes = number_classes
    #
    self.encoder = None
    self.decoder = None
    self.graph = None

  def encode(self,
             features,
             num_units,
             bottleneck_number_feature_maps=None):
    """
    Args:
    	inputs: image.
      scope: The scope of the block.
      bottleneck_number_feature_maps: If not None, 
        1x1 bottleneck reduces the output
      dropout_rate: dropout rate to apply for each conv unit in training.
      num_units: Array of number of units in each block.
    """
    rate = self.growth_rate
    bottleneck_maps = bottleneck_number_feature_maps
    with tf.variable_scope('encoder', 'densenet', [features], reuse=self.reuse):
      net = features
      initial_nb_layers = self.growth_rate * 2
      with tf.variable_scope("conv1"):
        net = tf.layers.conv2d(net,
                               initial_nb_layers,
                               use_bias=False,
                               kernel_size=7,
                               padding='same',
                               strides=1)
      # with tf.variable_scope("pool1"):
      #   net = tf.layers.max_pooling2d(net, [3, 3], strides=2, padding='same')
      # consecutive denseblocks each followed by a transition layer.
      for bn, num_units_in_block in enumerate(num_units):
        with tf.variable_scope("block_{}".format(bn + 1), values=[net]):
          for un, unit in enumerate(range(num_units_in_block)):
            with tf.variable_scope("unit_{}".format(un + 1), values=[net]):
              output = densenet_utils.stack_blocks_dense(net,
                                                         is_training=self.is_training,
                                                         growth_rate=rate,
                                                         bottleneck_number_feature_maps=bottleneck_maps,
                                                         dropout_keep_prob=self.dropout_keep_prob)

              net = tf.concat(axis=3, values=[net, output])
          # the last layer does not have a transition layer
          if bn + 1 != len(num_units):
            # the output of the dense block. add to the collection for upsampling
            # exclude the bottleneck (the last layer)
            tf.add_to_collection('skip_connections', net)
            # downsample
            net = densenet_utils.add_transition_down_layer(net, is_training=self.is_training)
      self.encoder = Encoder(graph=net.graph, output=net)

  def decode(self,
             num_units):
    encoder = self.encoder
    graph = encoder.graph
    with graph.as_default():
      with tf.variable_scope('decoder', 'densenet', reuse=self.reuse):
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
                                                           growth_rate=self.growth_rate,
                                                           dropout_keep_prob=self.dropout_keep_prob)
                # unlike downsampling path, do not concatenate the input layer within the dense block
                if un > 0:
                  net = tf.concat(axis=3, values=[net, output])
                else:
                  net = output
        with tf.variable_scope('logits'):
          net = tf.layers.conv2d(net,
                                 self.number_classes,
                                 kernel_size=1,
                                 activation=tf.identity)
    self.graph = net.graph
    return Decoder(graph=net.graph, output=net)

  def build_graph(self, image):
    k = 12
    num_encoder_units = [4, 5, 7, 10]
    bottleneck_number_feature_maps = k * 4
    self.encode(features=image,
                num_units=num_encoder_units,
                bottleneck_number_feature_maps=bottleneck_number_feature_maps)
    num_decoder_units = [10, 7, 5]
    self.decode(num_units=num_decoder_units)

def main(unused_argv):
  pass


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()




