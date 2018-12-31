import numpy as np 
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

__all__ = [
	'Rcnn_Base',
	'Weighted_Rcnn',
	'Rcnn_Prototype',
]

class Rcnn_Base(object):
	"""
	Med2vec + RCNN + full-connect + softmax
	"""
	def __init__(self, **config):
		'''	hyperparameter
			


		'''
		self.__dict__.update(config)
		self._build()
		self._open_session()

	def _build(self):
		### placeholder
		self._build_placeholder()
		### forward
		self._build_rcnn()


		### loss: classify loss

		self.classify_loss = 0

		### train_Op
		self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self.LR).minimize(self.classify_loss)


	def _build_placeholder(self):
		pass
		### embedding 
		### list of seq has equal length 


	def _build_rcnn(self):
		pass

	def _build_classify(self):
		pass






	def _open_session(self):
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())


class Weighted_Rcnn(Rcnn_Base):
	pass 


class Rcnn_Prototype(Rcnn_Base):
	pass 


if __name__ == '__main__':
	from config import get_RCNN_config
	config = get_RCNN_config()
	rcnn_base = Rcnn_Base(**config)


