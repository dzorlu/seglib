
import tensorflow as tf


def get_number_params(graph):
	with graph.as_default():
		total_parameters = 0
		for variable in tf.trainable_variables():
			# shape is an array of tf.Dimension
			shape = variable.get_shape()
			#print(shape)
			##print(len(shape))
			variable_parameters = 1
			for dim in shape:
			    #print(dim)
			    variable_parameters *= dim.value
			#print(variable_parameters)
			total_parameters += variable_parameters
			print("number of params in the model: {}".format(total_parameters))
