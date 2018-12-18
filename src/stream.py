import numpy as np 
from collections import defaultdict 
import torch
from torch import nn
from torch.autograd import Variable

#torch.manual_seed(3)
#np.random.seed(0)
### to do, zero-padding + embedding 

class Sequence_Data(object):
	def __init__(self, is_train = True, **config):
		super(Sequence_Data, self).__init__()
		self.max_length = config['max_length']
		self.num_class = config['num_class']
		self.is_train = is_train
		self.batch_size = config['batch_size']
		self.big_batch_size = config['big_batch_size']
		self.filename = config['train_file'] if self.is_train \
			else config['test_file']
		with open(self.filename, 'r') as fin:
			lines = fin.readlines()[1:]
		self.total_num = len(lines)
		self.label = list(map(lambda x:1 if x=='True' else  0, \
			[line.split('\t')[0] for line in lines]))
		self.sequence = list(map(lambda x: [int(i) for i in x.split()], \
			[line.split('\t')[2] for line in lines]))
		def f1(timestamp_str):
			return [int(i) for i in timestamp_str.split()]
		self.timestamp = list(map(f1, \
			[line.split('\t')[3] for line in lines]))

		#### cut-off max_length
		#self.sequence = self.sequence[-self.max_length:]
		self.sequence = [i[-self.max_length:] for i in self.sequence]
		self.sequence_len = [len(i) for i in self.sequence]
		#self.timestamp = self.timestamp[-self.max_length:]
		self.timestamp = [i[-self.max_length:] for i in self.timestamp]

		###  minus first  
		def f2(arr):
			return [i - arr[0] for i in arr]
		self.timestamp = list(map(f2, self.timestamp))

		### shuffle
		self.batch_id = 0
		self.batch_num = int(np.ceil(self.total_num / self.batch_size))
		self.random_shuffle = np.arange(self.total_num)

	def next(self, embedding):  ### to do, embedding file + zero-padding : return 1.label; 2.embedding; 3. leng;
		bgn, endn = self.batch_id * self.batch_size, (self.batch_id+1) * self.batch_size 
		self.batch_id += 1			
		if self.batch_id == self.batch_num:		
			self.batch_id = 0 ## random.shuffle 
			#print('epoch')
			#print(self.check_seq_leng-bgn, end = '======\n')
		seq = self.sequence[bgn:endn]
		for i in seq:
			i.extend([0] * (self.max_length - len(i)))
		seq = torch.LongTensor(seq)
		seq_embed = embedding(seq)
		#return self.label[bgn:endn], self.sequence[bgn:endn], self.timestamp[bgn:endn]
		return seq_embed, self.sequence_len[bgn:endn], self.label[bgn:endn]
		###		tensor, list,  list 

	def get_all(self, embedding):
		#seq_embed, _, __ = self.next(embedding)
		tmp = self.next(embedding)
		seq_embed = tmp[0]
		tmp0 = self.batch_id
		self.restart()
		_, x, y = seq_embed.shape
		Seq_embed = torch.FloatTensor(self.total_num, x, y)
		Seq_len = []
		for i in range(self.batch_num):
			##seq_embed, seq_len, _ = self.next(embedding)
			tmp = self.next(embedding)
			seq_embed, seq_len = tmp[0], tmp[1]
			Seq_embed[i * self.batch_size: (i+1)*self.batch_size] = seq_embed
			Seq_len.extend(seq_len)
		self.batch_id = tmp0 
		return Seq_embed, Seq_len

	def restart(self):
		self.batch_id = 0

	@property
	def batch_number(self):
		return self.batch_num

	@property
	def check_seq_leng(self):
		return len(self.sequence)

def embed_file_2_embed_mat(embed_file, word_size):
	with open(embed_file, 'r') as fin:
		lines = fin.readlines()[1:]
	embed_dim = len(lines[0].strip().split()) - 1
	assert embed_dim == 100 
	embed_mat = np.zeros((word_size, embed_dim), dtype = np.float)
	embed_dict = defaultdict(lambda: np.zeros(embed_dim))
	for line in lines:
		embed_dict[int(line.split()[0])] = np.array(line.strip().split()[1:]) 
	for i in range(word_size):
		embed_mat[i,:] = embed_dict[i]
	embed_mat = torch.FloatTensor(embed_mat)
	embedding = nn.Embedding.from_pretrained(embed_mat)
	return embedding


class Weight_Sequence_Data(Sequence_Data):		### for weight NN 
	def __init__(self, weight_lst = None, is_train = True, **config):
		Sequence_Data.__init__(self, is_train, **config)	
		if weight_lst == None:
			self.weight_lst = [1.0] * self.total_num
		else:
			self.weight_lst = weight_lst

	def update_weight(self, weight_lst):
		self.weight_lst = weight_lst

	def next(self, embedding):
		seq_weight = self.weight_lst[self.batch_id * self.batch_size: (self.batch_id + 1) * self.batch_size]
		seq_embed, seq_len, seq_label = Sequence_Data.next(self, embedding)
		return seq_embed, seq_len, seq_label, seq_weight


class Decision_Tree_Data(Weight_Sequence_Data):		### weight Decision Tree 
	def __init__(self, weight_lst = None, is_train = True, **config):
		Weight_Sequence_Data.__init__(self, is_train, **config)
		"""
		self.is_train = is_train
		self.filename = config['train_file'] if self.is_train \
			else config['test_file']

		self.total_num = len(lines)
		self.label = list(map(lambda x:1 if x=='True' else  0, \
			[line.split('\t')[0] for line in lines]))
		"""
		with open(self.filename, 'r') as fin:
			lines = fin.readlines()[1:]
		self.input_dim = config['admis_dim']		
		self.mat = np.zeros((self.total_num, self.input_dim))
		for i,line in enumerate(lines):
			arr = np.zeros((self.input_dim))
			for k in line.split('\t')[2].split():
				arr[int(k)] = 1
			self.mat[i] = arr 
		self.weight_lst = weight_lst if weight_lst != None else [1.0] * self.total_num

	### update_weight 		

class Post_Data(object):
	def __init__(self, is_train = True, **config):
		super(Post_Data, self).__init__()
		self.batch_size = config['batch_size']
		self.filename = config['new_train_file'] if is_train else config['new_test_file']
		data = np.load(self.filename)
		self.label = Variable(torch.LongTensor(data[:,0]))
		self.feature = Variable(torch.FloatTensor(data[:,1:]))

		self.total_num = data.shape[0]
		self.batch_num = int(np.ceil(self.total_num / self.batch_size))
		self.batch_id = 0


	def next(self):
		bgn, endn = self.batch_id * self.batch_size, (self.batch_id+1) * self.batch_size 
		self.batch_id += 1			
		if self.batch_id == self.batch_num:		
			self.batch_id = 0 
		return self.feature[bgn:endn], self.label[bgn:endn]

	def get_all(self):
		return self.feature, self.label		

class Weighted_Post_Data(Post_Data):
	def __init__(self, is_train = True, **config):
		Post_Data.__init__(self, is_train, **config)
		self.weight_lst = [1.0] * self.total_num

	def restart(self):
		self.batch_id = 0

	def next(self):
		bgn, endn = self.batch_id * self.batch_size, (self.batch_id+1) * self.batch_size 
		feat, lab = Post_Data.next(self)
		return feat, lab, self.weight_lst[bgn:endn]	

	@property
	def batch_number(self):
		return self.batch_num

	@property
	def weight(self):
		return self.weight_lst

	@staticmethod
	def normalize_weight(self):
		pass 

	def update_weight(self, weight_lst):
		self.weight_lst = weight_lst

	def get_all(self):
		feat, lab = Post_Data.get_all(self)
		return feat, lab, self.weight_lst


if __name__ == '__main__':
	from config import get_RCNN_config
	config = get_RCNN_config()
	embedding = embed_file_2_embed_mat(config['embed_file'], config['admis_dim'])
	#print(embedding)

	trainData = Sequence_Data(is_train = True, **config)
	#print(trainData.check_seq_leng)
	for i in range(config['train_iter']):
		seq_embed, seq_len, label = trainData.next(embedding)
		#print(seq_len)
		#print(i)
		#print(seq_len)
		#print(seq_embed.shape)
	pass









