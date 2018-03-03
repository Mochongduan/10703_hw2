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
class Conv:
	def _init_(self, output_classes: int, learning_rate, Input: tf.placeholder):
		self.output_classes = output_classes
		self.learning_rate = learning_rate
		self.input_data= tf.reshape(Input, [-1,128,1])
		
	def conv_model(self):
		conv1 = tf.layers.conv1d(self.input_data, 32, kernel_size=2, padding="same", activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2)

        conv2 = tf.layers.conv1d(pool1, 32, kernel_size=2, padding="same", activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=3, strides=2)

        conv3 = tf.layers.conv1d(pool2, 128, kernel_size=2, padding="same", activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2)
		
		flat_pool3 = tf.reshape(pool3, [-1, 16*128])
		
		layer = tf.layers.dense(pool3_flat, 512)
        layer = tf.layers.dense(layer, 128)
        layer = tf.layers.dense(layer, self.output_classes)

        self.inference = layer
        self.predict = tf.argmax(self.inference, 1)

        self.output_data= tf.placeholder(tf.float32, shape=[None, self.output_classes])
        self.loss = tf.losses.mean_squared_error(self.output_data, self.inference)

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.loss)

