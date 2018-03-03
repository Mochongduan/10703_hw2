import tensorflow as tf

class Conv:
	def _init_(self, classes: int, learning_rate, Input: tf.placeholder):
		self.classes = classes
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
		net = tf.layers.dense(pool3_flat, 512)
        	net = tf.layers.dense(net, 128)
        	net = tf.layers.dense(net, self.classes)

        	self.inference = net
        	self.predict = tf.argmax(self.inference, 1)

        	self.output_data= tf.placeholder(tf.float32, shape=[None, self.classes])
        	self.loss = tf.losses.mean_squared_error(self.output_data, self.inference)

        	self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
