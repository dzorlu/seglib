import tensorflow as tf


def get_dataset(x, y, batch_size=32, nb_epochs=None, shuffle=True):
	"""
	Create a tf.dataset from numpy arrays.
	:param x: 
	:param y: 
	:param batch_size: 
	:param nb_epochs: 
	:param shuffle: 
	:return: 
	"""
	with tf.name_scope('transform'):
		dataset = tf.data.Dataset.from_tensor_slices((x, y))
		if nb_epochs:
			dataset = dataset.repeat(nb_epochs)
		if shuffle:
			dataset = dataset.shuffle()
		dataset = dataset.batch(batch_size)
	return dataset


def get_iterator(output_types, output_shapes):
	"""
	Create a data iterator given output_types and output_shapes
	that is used as a reinitializable iterator
	:param output_types: 
	:param output_shapes: a (TensorShape, TensorShape) tupe 
	:return: 
	"""
	if type(output_shapes[0]) != tf.TensorShape or type(output_shapes[1]) != tf.TensorShape:
		raise ValueError("TensorShape not defined.")
	iterator = tf.data.Iterator.from_structure(output_types, output_shapes)
	return iterator


