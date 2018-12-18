import torch
from torch import nn 
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np 
#torch.manual_seed(2)
#np.random.seed(0)

class RCNN(nn.Module):
	"""
		baseline RCNN
	"""

	def __init__(self, **config):
		super(RCNN, self).__init__()

		#### CNN
		self.cnn_in_channel = config['embed_dim']
		self.cnn_kernel_size = config['cnn_kernel_size']
		self.cnn_stride = config['cnn_stride']
		self.cnn_out_channel = config['cnn_out_channel']
		self.maxpool_size = config['maxpool_size']
		self.conv1 = nn.Conv1d(in_channels = self.cnn_in_channel, out_channels = self.cnn_out_channel, \
			kernel_size = self.cnn_kernel_size, stride = self.cnn_stride)
		self.maxpool = nn.MaxPool1d(kernel_size = self.maxpool_size)

		### RNN 
		self.rnn_input_size = self.cnn_out_channel #####
		self.rnn_hidden_size = config['rnn_hidden_size']
		self.rnn_num_layer = config['rnn_num_layer']
		self.batch_first = config['batch_first']
		self.bidirectional = config['bidirectional']
		self.rnn = nn.LSTM(
            input_size = self.rnn_input_size, 
            hidden_size = int(self.rnn_hidden_size / 2),
            num_layers = self.rnn_num_layer,
            batch_first = self.batch_first,
            bidirectional= self.bidirectional
            )

		#### linear
		self.num_class = config['num_class']
		self.out = nn.Linear(self.rnn_hidden_size, self.num_class)
		self.loss = nn.CrossEntropyLoss()

		self.batch_size = config['batch_size']
		#### prototype
		#self.prototype_vec = Variable(torch.zeros( ,  ))

	def forward_cnn(self, X_embed, X_len):
		"""
			X_embed is tensor:  batch_size, max_length, embedding_size
			X_len: is list 
		"""
		X_embed = Variable(X_embed)
		X_embed = X_embed.permute(0,2,1)	####   batch_size, embedding_size, max_length
		X_conv = self.conv1(X_embed)
		X_maxpool = self.maxpool(X_conv)
		X_maxpool = X_maxpool.permute(0,2,1)

		f_leng = lambda x: max(int((int((x - self.cnn_kernel_size) / self.cnn_stride) + 1) / self.maxpool_size), 1)
		X_len = list(map(f_leng, X_len))
		return X_maxpool, X_len

	def forward_rnn(self, X, X_len):
		dd1 = sorted(range(len(X_len)), key=lambda k: X_len[k], reverse = True)
		dd = [0 for i in range(len(dd1))]
		for i,j in enumerate(dd1):
			dd[j] = i
		X_len_sort = list(np.array(X_len)[dd1])
		X_v = X[dd1]
		pack_X_batch = torch.nn.utils.rnn.pack_padded_sequence(X_v, X_len_sort, batch_first=True)
		_,(X_out,_) = self.rnn(pack_X_batch, None)
		X_out2 = torch.cat([X_out[0], X_out[1]], 1)
		X_out2 = X_out2[dd]
		return X_out2

	def forward_rcnn(self, X_embed, X_len):
		X1, X_len1 = self.forward_cnn(X_embed, X_len)
		X2 = self.forward_rnn(X1, X_len1)
		return X2		

	def forward(self, X_embed, X_len, label):
		### processing label?
		label = Variable(torch.LongTensor(label))
		### NN-level  
		#X1, X_len1 = self.forward_cnn(X_embed, X_len)
		#X2 = self.forward_rnn(X1, X_len1)
		X2 = self.forward_rcnn(X_embed, X_len)
		X3 = self.out(X2)

		### loss-level
		loss_crossentropy = self.loss(X3, label)
		return loss_crossentropy, X3 


class Prototype_RCNN(RCNN):
	"""
		using cosine distance in prototype layer
	"""
	def __init__(self, assignment, **config):
		RCNN.__init__(self, **config)
		self.prototype_num = len(assignment)
		self.assignment = assignment 
		self.out = nn.Linear(self.prototype_num, self.num_class)
		self.prototype_vec = Variable(torch.zeros(self.prototype_num, self.rnn_hidden_size), requires_grad = False)

	@staticmethod
	def normalize_by_column(T_2d):
		return T_2d / T_2d.norm(dim = 1, keepdim = True)

	def generate_single_prototype(self, X_in, X_len):
		bs = X_in.shape[0]
		X_out = torch.zeros(bs, self.rnn_hidden_size)
		for i in range(int(np.ceil(bs / self.batch_size))):
			bgn, endn = i * self.batch_size, (i+1) * self.batch_size
			X_out[bgn:endn] = self.forward_rcnn(X_in[bgn:endn], X_len[bgn:endn])
		X_mean = X_out.mean(0)
		return X_mean.data

	def generate_prototype(self, X_in_all, X_len_all):
		prototype_vector = torch.zeros(self.prototype_num, self.rnn_hidden_size)
		for i,j in enumerate(self.assignment):
			X_in = X_in_all[j]
			X_len = [X_len_all[k] for k in j]
			prototype_vector[i,:] = self.generate_single_prototype(X_in, X_len)  
		prototype_vector = self.normalize_by_column(prototype_vector)
		self.prototype_vec.data = prototype_vector

	def forward_prototype(self, X):
		X_norm = self.normalize_by_column(X)
		return X_norm.matmul(self.prototype_vec.transpose(1,0))

	def forward(self, X_embed, X_len, label):
		label = Variable(torch.LongTensor(label))	
		X2 = self.forward_rcnn(X_embed, X_len)		
		X3 = self.forward_prototype(X2)
		X3 = self.out(X3)

		### loss-level
		loss_crossentropy = self.loss(X3, label)
		return loss_crossentropy, X3 

class Prototype_RCNN_L2(Prototype_RCNN):
	"""
		using L2 distance in prototype layer
	"""
	def generate_prototype(self, X_in_all, X_len_all):
		prototype_vector = torch.zeros(self.prototype_num, self.rnn_hidden_size)
		for i,j in enumerate(self.assignment):
			X_in = X_in_all[j]
			X_len = [X_len_all[k] for k in j]
			prototype_vector[i,:] = self.generate_single_prototype(X_in, X_len)  
		self.prototype_vec.data = prototype_vector

	def forward_prototype(self, X):
		bs = X.shape[0]
		### 1. prototype vector
		prototype_vec_3d = self.prototype_vec.view(1, self.prototype_num, self.rnn_hidden_size)
		prototype_vec_3d = prototype_vec_3d.expand(bs, self.prototype_num, self.rnn_hidden_size)
		### 2. 
		X_3d = X.view(bs, self.rnn_hidden_size, 1)
		X_3d = X_3d.expand(bs, self.rnn_hidden_size, self.prototype_num)
		X_3d = X_3d.permute(0,2,1)
		X_diff = (prototype_vec_3d - X_3d)**2
		X_sum = X_diff.sum(2)
		return X_sum

class Prototype_RCNN_weighted(Prototype_RCNN_L2): ### Prototype_RCNN, Prototype_RCNN_L2

	def forward(self, X_embed, X_len, label, weight = None):
		label = Variable(torch.LongTensor(label))
		X2 = self.forward_rcnn(X_embed, X_len)
		X3 = self.forward_prototype(X2)
		X3 = self.out(X3)
		#weight_t = torch.FloatTensor(weight).diag()
		batch_size = X3.shape[0]
		if weight == None:
			weight = [1.0] * batch_size
		for i in range(batch_size):
			loss_crossentropy = self.loss(X3[i].reshape(1,-1), label[i].reshape(1)) * weight[i] if i == 0 \
				else loss_crossentropy + self.loss(X3[i].reshape(1,-1), label[i].reshape(1)) * weight[i]
		#loss_crossentropy = self.loss(X3, label)
		return loss_crossentropy, X3


class Post_Prototype_RCNN_L2(Prototype_RCNN):	### Prototype_RCNN_L2, Prototype_RCNN

	def __init__(self, assignment, **config):
		Prototype_RCNN.__init__(self, assignment, **config)
		### to do: highway 

	def generate_single_prototype(self, X):
		return X.mean(0)

	def generate_prototype(self, X_in_all):
		print('____call_____')
		prototype_vector = torch.zeros(self.prototype_num, self.rnn_hidden_size)
		for i,j in enumerate(self.assignment):
			X_in = X_in_all[j]
			prototype_vector[i,:] = self.generate_single_prototype(X_in) 
		self.prototype_vec.data = prototype_vector

	def forward(self, X, label):
		X1 = self.forward_prototype(X)
		X2 = self.out(X1)
		loss_crossentropy = self.loss(X2, label)

		'''
		batch_size = X2.shape[0]
		weight = [1.0] * batch_size
		for i in range(batch_size):
			loss_crossentropy = self.loss(X2[i].reshape(1,-1), label[i].reshape(1)) * weight[i] if i == 0 \
				else loss_crossentropy + self.loss(X2[i].reshape(1,-1), label[i].reshape(1)) * weight[i]
		loss_crossentropy = loss_crossentropy / batch_size
		'''
		return loss_crossentropy, X2

class Weighted_Post_Prototype_RCNN_L2(Post_Prototype_RCNN_L2):
	def __init__(self, assignment, **config):
		Post_Prototype_RCNN_L2.__init__(self, assignment, **config)
	
	def generate_single_prototype(self, X, weight):
		assert len(weight) == X.shape[0]
		weight_sum = sum(weight)
		weight = torch.FloatTensor(weight).diag()
		X = weight.matmul(X)
		return X.sum(0) / weight_sum

	def generate_prototype(self, X_in_all, weight):
		prototype_vector = torch.zeros(self.prototype_num, self.rnn_hidden_size)
		for i,j in enumerate(self.assignment):
			X_in = X_in_all[j]
			weight0 = [weight[k] for k in j]
			prototype_vector[i,:] = self.generate_single_prototype(X_in, weight0) 
		self.prototype_vec.data = prototype_vector
	
	def forward(self, X, label, weight = None):
		X1 = self.forward_prototype(X)
		X2 = self.out(X1)
		batch_size = X2.shape[0]
		if weight == None:
			loss_crossentropy = self.loss(X2, label)
			return loss_crossentropy, X2 
			#weight = [1.0] * batch_size
		for i in range(batch_size):
			loss_crossentropy = self.loss(X2[i].reshape(1,-1), label[i].reshape(1)) * weight[i] if i == 0 \
				else loss_crossentropy + self.loss(X2[i].reshape(1,-1), label[i].reshape(1)) * weight[i]
		loss_crossentropy = loss_crossentropy / batch_size
		return loss_crossentropy, X2

	def measure_similarity(self, X, weight = None):
		X1 = self.forward_prototype(X)
		
		weight = X1.sum(1)	### bs, 1
		

		return list(weight.data.numpy())








if __name__ == '__main__':
	from config import get_RCNN_config
	config = get_RCNN_config()
	rcnn = RCNN(**config)





