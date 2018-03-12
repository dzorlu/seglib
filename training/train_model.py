import tensorflow as tf
import argparse
import sys

sys.path.append('/Users/denizzorlu/seglib/')


from models.densenet import DenseNet
from data.data_generator import generate_data
from utils.tf_utils import get_number_params


FLAGS = tf.flags.FLAGS
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=4, type=int, help='batch size')
parser.add_argument('--number_classes', default=2, type=int, help='number of classes')
parser.add_argument('--model_dir', default='/tmp/tf/seg/', type=str, help='number of classes')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

# Learning hyperaparmeters
_BASE_LR = 0.1
_LR_SCHEDULE = [  # (LR multiplier, epoch to start)
    (1.0 / 6, 0), (2.0 / 6, 1), (3.0 / 6, 2), (4.0 / 6, 3), (5.0 / 6, 4),
    (1.0, 5), (0.1, 30), (0.01, 60), (0.001, 80), (0.0001, 90)
]


def add_summaries(graph, predictions, labels):
	# Add summaries for images, variables and losses.
	global_summaries = set([])
	# prediction summary
	predictions = tf.expand_dims(predictions, 3)
	predictions = tf.cast(predictions, tf.float32)
	predictions = tf.multiply(predictions, 255)
	image_summary = tf.summary.image('image_summary', predictions)
	global_summaries.add(image_summary)
	for model_var in tf.get_collection('trainable_variables'):
		global_summaries.add(tf.summary.histogram(model_var.op.name, model_var))
		# total loss
	summary_op = tf.summary.merge(list(global_summaries), name='summary_op')
	return summary_op


def learning_rate_schedule(current_epoch):
  """Handles linear scaling rule, gradual warmup, and LR decay."""
  scaled_lr = _BASE_LR * (FLAGS.train_batch_size / 256.0)

  decay_rate = scaled_lr
  for mult, start_epoch in _LR_SCHEDULE:
    decay_rate = tf.where(current_epoch < start_epoch, decay_rate,
                          scaled_lr * mult)

  return decay_rate


def model_fn(features, labels, mode, params):
	"""Model function for DenseNet classifier.
	
	Args:
	  features: inputs.
	  labels: one hot encoded classes
	  mode: one of tf.estimator.ModeKeys.{TRAIN, INFER, EVAL}
	  params: a parameter dictionary with the following keys: 
	
	Returns:
	  ModelFnOps for Estimator API.
	"""
	number_classes = params.get('number_classes')
	growth_rate = params.get('growth_rate')
	dropout_keep_prob = params.get('dropout_keep_prob')
	encoder_num_units = params.get('encoder_num_units')
	decoder_num_units = params.get('decoder_num_units')
	bottleneck_number_feature_maps = params.get('bottleneck_number_feature_maps')

	densenet = DenseNet(growth_rate=growth_rate,
											dropout_keep_prob=dropout_keep_prob,
											number_classes=number_classes,
											is_training=(mode == tf.estimator.ModeKeys.TRAIN))
	densenet.encode(features=features,
									num_units=encoder_num_units,
									bottleneck_number_feature_maps=bottleneck_number_feature_maps)
	logits = densenet.decode(decoder_num_units).output
	probs = tf.nn.softmax(logits)

	predictions = tf.argmax(logits, axis=-1)
	if mode == tf.estimator.ModeKeys.PREDICT:
		predictions = {
			'class_ids': predictions,
			'probabilities': probs,
			'logits': logits,
		}
		return tf.estimator.EstimatorSpec(mode, predictions=predictions)

	# Add the loss.
	# Calculate loss, which includes softmax cross entropy and L2 regularization.
	# wraps the softmax_with_entropy fn. adds it to loss collection
	tf.losses.softmax_cross_entropy(
		logits=logits, onehot_labels=labels)
	# include the regulization losses in the loss collection.
	loss = tf.losses.get_total_loss()
	# tf.logging.info("Starting to evaluate..")
	# if mode == tf.estimator.ModeKeys.EVAL:
	# 	tf.logging.info("Starting to evaluate..")
	# 	return tf.estimator.EstimatorSpec(
	# 		mode=mode,
	# 		eval_metric_ops={"accuracy": tf.metrics.accuracy(labels, predictions)})

	assert mode == tf.estimator.ModeKeys.TRAIN
	tf.logging.info("Starting to train..")
	global_step = tf.train.get_global_step()
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	optimizer = tf.train.AdagradOptimizer(learning_rate=0.001)
	# https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization
	with tf.control_dependencies(update_ops):
		train_op = optimizer.minimize(loss, global_step=global_step)

	add_summaries(densenet.graph, predictions, labels)
	return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def main(argv):
	args = parser.parse_args(argv[1:])

	config = tf.estimator.RunConfig(save_checkpoints_steps=30,
	                                save_summary_steps=30,
	                                model_dir=args.model_dir)
	classifier = tf.estimator.Estimator(
		model_fn=model_fn,
		config=config,
		params={
			'number_classes': args.number_classes,
			'growth_rate': 12,
			'dropout_keep_prob': 0.2,
			'encoder_num_units': [4, 5, 7, 10],
			'decoder_num_units': [10, 7, 5],
			'bottleneck_number_feature_maps': 48
		}
	)

	# Train the Model.
	classifier.train(
		input_fn=lambda: generate_data(args.batch_size, args.number_classes),
		steps=args.train_steps)

	# Predict the model.
	pred_result = classifier.predict(
		input_fn=lambda: generate_data(args.batch_size, args.number_classes, inference=True))
	print(pred_result)


if __name__ == '__main__':
	tf.logging.set_verbosity(tf.logging.INFO)
	tf.app.run(main)

