import tensorflow as tf

class MLP:
	def __init__(self, Input: tf.placeholder, output_classes: int, learning_rate):
	    length_of_state = Input.get_shape().as_list()[1]
	    self.input_data = tf.reshape(Input, [-1, length_of_state])
		self.output_classes = output_classes
	    self.learning_rate = learning_rate
	def mlp_3_hidden(self):
	        layer = self.input_data
	        layer = tf.layers.dense(layer, 16, activation=tf.nn.relu)
	        layer = tf.layers.dense(layer, 16, activation=tf.nn.relu)
	        layer = tf.layers.dense(layer, 16, activation=tf.nn.relu)
	        layer = tf.layers.dense(layer, self.output_classes)
	        self.inference = layer
	        self.predict = tf.argmax(self.inference, 1)
	        self.output = tf.placeholder(tf.float32, shape=[None, self.output_classes])
	        self.loss = tf.losses.mean_squared_error(self.output, self.inference)
