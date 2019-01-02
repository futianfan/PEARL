import numpy as np 
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from collections import defaultdict 

__all__ = [
	'Rcnn_Base',
	'Weighted_Rcnn',
	'Weighted_Rcnn_Prototype',
]


def _1dlabel_to_2dlabel(batch_label):
	batch_size = len(batch_label)
	label_2d = np.zeros((batch_size, 2),dtype = int)
	for i in range(batch_size):
		label_2d[i, int(batch_label[i])] = 1
	return label_2d

def _embed_file_2_numpy_embed_mat(embed_file, word_size):
	with open(embed_file, 'r') as fin:
		lines = fin.readlines()[1:]
	embed_dim = len(lines[0].strip().split()) - 1
	assert embed_dim == 100 
	embed_mat = np.zeros((word_size, embed_dim), dtype = np.float32)
	embed_dict = defaultdict(lambda: np.zeros(embed_dim))
	for line in lines:
		embed_dict[int(line.split()[0])] = np.array(line.strip().split()[1:]) 
	for i in range(word_size):
		embed_mat[i,:] = embed_dict[i]
	return embed_mat


class Rcnn_Base(object):
	"""
	Med2vec + RCNN + full-connect + softmax
	"""
	def __init__(self, **config):
		'''	###  hyperparameter
				config['new_train_file'] = os.path.join(config['HF_folder'], 'train.npy')
				config['new_test_file'] = os.path.join(config['HF_folder'], 'test.npy')

				config['embed_dim'] = 100
				config['admis_dim'] = 1867
				config['max_length'] = 50
				config['num_class'] = 2
				config['batch_size'] = 512  ### 32, 256 
				config['big_batch_size'] = 256
				config['train_iter'] = 120 ## int(2e3)

				### Neural Network
				config['cnn_kernel_size'] = 10
				config['cnn_stride'] = 1 
				config['cnn_out_channel'] = 150
				config['maxpool_size'] = 3
				config['rnn_hidden_size'] = 50 
				config['rnn_num_layer'] = 1
				config['batch_first']  = True
				config['bidirectional'] = True

				config['num_class'] = 2
				config['LR'] = 1e-1		### 1e-1
		'''
		self.__dict__.update(config)
		self.embed_mat = _embed_file_2_numpy_embed_mat(config['embed_file'], config['admis_dim'])
		self._build()
		self._open_session()

	def _build(self):
		### placeholder
		self._build_placeholder()

		### forward: RCNN
		self._build_rcnn()

		### full-connect for classify
		self._build_classify()

		### train_Op
		self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self.LR).minimize(self.classify_loss)

	def _build_placeholder(self):
		self.seqlst = tf.placeholder(dtype = tf.int32, shape = [None, self.max_length])
		self.X = tf.nn.embedding_lookup(params = self.embed_mat, ids = self.seqlst)
		assert self.X.dtype == tf.float32		
		self.seqlen = tf.placeholder(dtype = tf.int32, shape = [None])
		self.y = tf.placeholder(dtype = tf.int32, shape = [None, self.num_class])
		### embedding 
		### list of seq has equal length 

	def _build_rcnn(self):
		### CNN conv1d
		self.conv  = tf.layers.conv1d(
			inputs = self.X, 
			filters = self.cnn_out_channel, 
			kernel_size = self.cnn_kernel_size, 
			strides = self.cnn_stride, 
			padding = 'same'   #### same , valid 
			)
		### CNN max-pooling 
		self.mp = tf.layers.max_pooling1d(
			inputs = self.conv, 
			pool_size = self.maxpool_size, 
			strides = self.maxpool_stride, 
			padding = 'same'     #### same , valid
			) 

		### seq_len
		##  self.seqlen =>  seq_len_rnn
		#f = lambda x: max(self.max_length - self.cnn_kernel_size + 1 - self.maxpool_size + 1, 1)
		#self.seq_len_rnn = list(map(f, self.seqlen))
		### RNN part
		batch_size = tf.shape(self.X)[0] 
		self.X_ = tf.unstack(value = self.X, num = self.max_length, axis = 1)
		assert len(self.X_) == self.max_length
		lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units = self.rnn_hidden_size)
		outputs, state = tf.contrib.rnn.static_rnn(inputs = self.X_, cell = lstm_cell,\
		 dtype = tf.float32, sequence_length = self.seqlen)
		outputs = tf.stack(outputs, axis = 1)
		index = tf.range(0, batch_size) * self.max_length + (self.seqlen - 1)
		self.rnn_outputs = tf.gather(tf.reshape(outputs, [-1, self.rnn_hidden_size]), index)


	def _build_classify(self):
		weight_fc = tf.Variable(tf.random_normal(shape = [self.rnn_hidden_size, self.num_class]), dtype = tf.float32)
		bias_fc = tf.Variable(tf.zeros(shape = self.num_class), dtype = tf.float32)
		self.output_logits = tf.matmul(self.rnn_outputs, weight_fc) + bias_fc
		self.output_softmax = tf.nn.softmax(self.output_logits, axis = 1)
		self.classify_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
										labels=self.y, 
										logits=self.output_logits))

	def _open_session(self):
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

	def train(self, X, Y_1d, seqlen):
		Y_2d = _1dlabel_to_2dlabel(Y_1d)
		loss, _ = self.sess.run([self.classify_loss, self.train_op], \
			feed_dict = {self.seqlst:X, self.y:Y_2d, self.seqlen:seqlen})
		return loss 

	def evaluate(self, X, seqlen):
		return self.sess.run([self.output_softmax], \
			feed_dict = {self.seqlst:X, self.seqlen:seqlen})






class Weighted_Rcnn(Rcnn_Base):
	"""
	weighted, prototype, prototype_loss
	""" 
	def __init__(self, **config):
		self.__dict__.update(config)
		self.embed_mat = _embed_file_2_numpy_embed_mat(config['embed_file'], config['admis_dim'])
		self._build()
		self._open_session()		

	def _build(self):
		### placeholder
		self._build_placeholder()
		### forward: RCNN
		self._build_rcnn()
		### full-connect for classify
		self._build_classify()
		### train_Op
		self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self.LR).minimize(self.classify_loss)

	def _build_placeholder(self):
		Rcnn_Base._build_placeholder(self)
		self.weight = tf.placeholder(dtype = tf.float32, shape = [None])

	def _build_classify(self):
		weight_fc = tf.Variable(tf.random_normal(shape = [self.rnn_hidden_size, self.num_class]), dtype = tf.float32)
		bias_fc = tf.Variable(tf.zeros(shape = self.num_class), dtype = tf.float32)
		self.output_logits = tf.matmul(self.rnn_outputs, weight_fc) + bias_fc
		self.output_softmax = tf.nn.softmax(self.output_logits, axis = 1)
		self.classify_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
										labels=self.y, 
										logits=self.output_logits) \
										* self.weight \
		)

	def train(self, X, Y_1d, seqlen, weight):
		Y_2d = _1dlabel_to_2dlabel(Y_1d)
		loss, _ = self.sess.run([self.classify_loss, self.train_op], \
			feed_dict = {self.seqlst:X, self.y:Y_2d, self.seqlen:seqlen, self.weight:weight})
		return loss 

	def evaluate(self, X, seqlen):
		batch_size = len(seqlen)
		weight = [1.0] * batch_size
		return self.sess.run([self.output_softmax], \
			feed_dict = {self.seqlst:X, self.seqlen:seqlen, self.weight:weight})		



class Weighted_Rcnn_Prototype(Weighted_Rcnn):

	"""
	config['weight_file'] = ''
	config['eta'] = 1e-3   #### prototype loss 
	######

	config['new_train_file'] = os.path.join(config['HF_folder'], 'train.npy')
	config['new_test_file'] = os.path.join(config['HF_folder'], 'test.npy')

	config['embed_dim'] = 100
	config['admis_dim'] = 1867
	config['max_length'] = 50
	config['num_class'] = 2
	config['batch_size'] = 8  ### 32, 256 
	config['big_batch_size'] = 256
	config['train_iter'] = 30000 ## int(2e3)

	### Neural Network
	config['cnn_kernel_size'] = 10  ### 10 => 15 improvement  
	config['cnn_stride'] = 1 
	config['cnn_out_channel'] = 150
	config['maxpool_size'] = 9
	config['maxpool_stride'] = 1
	config['rnn_hidden_size'] = 50 
	config['rnn_num_layer'] = 1
	config['batch_first']  = True
	config['bidirectional'] = True

	config['LR'] = 1e-1		### 1e-1
	"""
	def __init__(self, assignment, **config):
		self.__dict__.update(config)
		self.embed_mat = _embed_file_2_numpy_embed_mat(config['embed_file'], config['admis_dim'])
		### assignment
		self.assignment = assignment
		self.prototype_num = len(assignment)
		### assignment
		self._build()
		self._open_session()		

	@staticmethod
	def _prototype_layer(X_, prototype_vector_):
		"""
			X_: b, d   batch_size  , dim   ***** b is None owing to ***placeholder****
			prototype_vector_: p, d   prototype_num, dim
		"""
		prototype_num = prototype_vector_.get_shape()[0]
		for i in range(prototype_num):
			y = X_ - tf.gather(prototype_vector_, [i])
			#y = X_ - tf.gather(prototype_vector_, tf.Variable([i]))		
			y = tf.norm(y, ord = 'euclidean', axis = 1)
			y = tf.reshape(y, [1, -1])
			if i == 0:
				output = y 
			else:
				output = tf.concat([output, y], 0)
		output = tf.transpose(output, perm = [1,0])  ### p,b => b,p
		return output

	def generate_prototype(self):
		"""
			TO DO list
		"""
		self.prototype_vector_ = tf.Variable(tf.random_normal(shape = [self.prototype_num, self.rnn_hidden_size]))
		### freeze the gradient. 
		self.prototype_vector_ = tf.stop_gradient(self.prototype_vector_)

	def _build_prototype(self):
		self.prototype_output = self._prototype_layer(self.rnn_outputs, self.prototype_vector_)
		"""
			TO DO LIST
		"""
		self.prototype_loss = tf.Variable(0.0, trainable = False) 


	def _build_classify(self):
		weight_fc = tf.Variable(tf.random_normal(shape = [self.prototype_num, self.num_class]), dtype = tf.float32)
		bias_fc = tf.Variable(tf.zeros(shape = self.num_class), dtype = tf.float32)
		self.output_logits = tf.matmul(self.prototype_output, weight_fc) + bias_fc
		self.output_softmax = tf.nn.softmax(self.output_logits, axis = 1)
		self.classify_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
										labels=self.y, 
										logits=self.output_logits) \
										* self.weight \
		)
		

	def _build(self):
		self._build_placeholder()
		self._build_rcnn()
		self.generate_prototype()
		self._build_prototype()
		self._build_classify()

		self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self.LR).\
						minimize(self.classify_loss + self.eta * self.prototype_loss)

	def train(self, X, Y_1d, seqlen, weight):
		Y_2d = _1dlabel_to_2dlabel(Y_1d)
		loss, _ = self.sess.run([self.classify_loss, self.train_op], \
			feed_dict = {self.seqlst:X, self.y:Y_2d, self.seqlen:seqlen, self.weight:weight})
		return loss 		

	'''
	### evaluate function use *****Weighted_Rcnn.evaluate******
	def evaluate(self, X, seqlen):
		batch_size = len(seqlen)
		weight = [1.0] * batch_size
		return self.sess.run([self.output_softmax], \
			feed_dict = {self.seqlst:X, self.seqlen:seqlen, self.weight:weight})		
	'''



if __name__ == '__main__':
	from config import get_RCNN_config
	config = get_RCNN_config()
	rcnn_base = Rcnn_Base(**config)


