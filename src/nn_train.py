import torch
from torch import nn 
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np 
from sklearn.metrics import roc_auc_score
from time import time
torch.manual_seed(4)
np.random.seed(3)
## torch 3, np 1
def evaluate(rnn_model, testData, embedding):
	testData.restart()
	Label = []
	Output = []
	for i in range(testData.batch_number):
		#seq_embed, seq_len, label = testData.next(embedding)
		tmp = testData.next(embedding)
		seq_embed, seq_len, label = tmp[0], tmp[1], tmp[2]
		_, output = rnn_model(seq_embed, seq_len, label)
		Label.extend(label)
		Output.extend([float(output[j][1]) for j in range(output.shape[0])])
	return roc_auc_score(Label, Output)	


def train_RCNN():
	from config import get_RCNN_config
	from nn_model import RCNN 
	from stream import embed_file_2_embed_mat, Sequence_Data
	config = get_RCNN_config()

	#### data prepare 
	embedding = embed_file_2_embed_mat(config['embed_file'], config['admis_dim'])
	trainData = Sequence_Data(is_train = True, **config)
	testData = Sequence_Data(is_train = False, **config)
	
	#### model & train 
	rcnn = RCNN(**config)
	LR = config['LR']
	opt_  = torch.optim.SGD(rcnn.parameters(), lr=LR)


	for i in range(config['train_iter']):
		seq_embed, seq_len, label = trainData.next(embedding)
		loss, _ = rcnn(seq_embed, seq_len, label)
		opt_.zero_grad()
		loss.backward()
		opt_.step()
		loss_value = loss.data[0]
		#print(loss_value)
		if i % 1 == 0:
			score = evaluate(rcnn, testData, embedding)
			print('{}-th iteration: auc: {}'.format(i, str(score)[:6]), end = ' ')
			if i > 0:
				print('cost {} seconds'.format(str(time() - t1)[:4]))
			t1 = time()

	#### save data 
	trainData.restart()
	trainMat = np.zeros((trainData.total_num, rcnn.rnn_hidden_size + 1),dtype = np.float)
	trainLab = []
	for i in range(trainData.batch_number):
		seq_embed, seq_len, label = trainData.next(embedding)
		trainLab.extend(label)
		X = rcnn.forward_rcnn(seq_embed, seq_len)
		trainMat[i*trainData.batch_size: (i+1)*trainData.batch_size,\
			1:] = X.data.numpy()
	trainMat[:,0] = np.array(trainLab)
	np.save(config['new_train_file'], trainMat)
	
	testData.restart()
	testMat = np.zeros((testData.total_num, rcnn.rnn_hidden_size + 1), dtype = np.float)
	testLab = []
	for i in range(testData.batch_number):
		seq_embed, seq_len, label = testData.next(embedding)
		testLab.extend(label)
		X = rcnn.forward_rcnn(seq_embed, seq_len)
		testMat[i*testData.batch_size:(i+1)*testData.batch_size,\
		1:] = X.data.numpy()
	testMat[:,0] = np.array(testLab)
	np.save(config['new_test_file'], testMat)	



def train_Prototype_RCNN(assignment = None):
	from config import get_RCNN_config
	from nn_model import Prototype_RCNN , Prototype_RCNN_L2
	from stream import embed_file_2_embed_mat, Sequence_Data
	config = get_RCNN_config()
	#### data prepare 
	embedding = embed_file_2_embed_mat(config['embed_file'], config['admis_dim'])
	trainData = Sequence_Data(is_train = True, **config)
	testData = Sequence_Data(is_train = False, **config)
	#### model & train 
	best_auc = 0
	gap = 300
	#if assignment == None:
	#	assignment = [list(range(i * gap, (i+1) * gap)) for i in range(30)]
	rcnn = Prototype_RCNN_L2(assignment, **config)
	LR = config['LR']
	opt_  = torch.optim.SGD(rcnn.parameters(), lr=LR)

	Seq_embed, Seq_len = trainData.get_all(embedding)

	for i in range(config['train_iter']):
		#print('{}-th iteration'.format(i))
		#t1 = time()
		rcnn.generate_prototype(Seq_embed, Seq_len)
		#print('cost {} seconds'.format(time() - t1))
		seq_embed, seq_len, label = trainData.next(embedding)
		loss, _ = rcnn(seq_embed, seq_len, label)
		opt_.zero_grad()
		loss.backward()
		opt_.step()
		loss_value = loss.data[0]
		#print(loss_value)
		if i % 1 == 0:
			score = evaluate(rcnn, testData, embedding)
			best_auc = score if score > best_auc else best_auc
			print('iter {}, auc:{} (best:{})'.format(i, str(score)[:5], str(best_auc)[:5]), end = ' ')
			if i > 0:
				print('{} sec'.format(int(time() - t1)))
			t1 = time()


def decision_tree_learning():
	from decision_tree import DT_learning
	assignment = DT_learning()
	return assignment


def train_weighted_Prototype_RCNN(assignment = None):
	from config import get_RCNN_config
	from nn_model import Prototype_RCNN_weighted
	from stream import embed_file_2_embed_mat, Weight_Sequence_Data

	config = get_RCNN_config()
	#### data prepare 
	embedding = embed_file_2_embed_mat(config['embed_file'], config['admis_dim'])
	trainData = Weight_Sequence_Data(is_train = True, **config)
	testData = Weight_Sequence_Data(is_train = False, **config)
	#### model & train 
	gap = 300
	best_auc = 0
	#if assignment == None:
	#	assignment = [list(range(i * gap, (i+1) * gap)) for i in range(30)]

	rcnn = Prototype_RCNN_weighted(assignment, **config)
	LR = config['LR']
	opt_  = torch.optim.SGD(rcnn.parameters(), lr=LR)
	Seq_embed, Seq_len = trainData.get_all(embedding)

	for i in range(config['train_iter']):
		#print('{}-th iteration'.format(i))
		rcnn.generate_prototype(Seq_embed, Seq_len)
		#print('cost {} seconds'.format(time() - t1))
		seq_embed, seq_len, label, weight = trainData.next(embedding)
		loss, _ = rcnn(seq_embed, seq_len, label, weight)
		opt_.zero_grad()
		loss.backward()
		opt_.step()
		loss_value = loss.data[0]
		#print(loss_value)
		if i % 1 == 0:
			score = evaluate(rcnn, testData, embedding)
			best_auc = score if score > best_auc else best_auc
			print('iter {}, auc:{} (best:{})'.format(i, str(score)[:5], str(best_auc)[:5]), end = ' ')
			if i > 0:
				print('{} sec'.format(int(time() - t1)))
			t1 = time()


def post_evaluate(testFeature, testLabel, nn, **config):
	batch_size = config['batch_size']
	batch_number = int(np.ceil(testFeature.shape[0] / batch_size))
	Label = []
	Output = []
	for i in range(batch_number):
		feat, label = testFeature[i*batch_size:(i+1)*batch_size], testLabel[i*batch_size:(i+1)*batch_size]
		_, output = nn(feat, label)
		label = list(label.data)
		Label.extend(label)
		Output.extend([float(output[j][1]) for j in range(output.shape[0])])
	return roc_auc_score(Label, Output)


def train_post_prototype(assignment):
	from config import get_prototype_config
	from nn_model import Post_Prototype_RCNN_L2
	from stream import Post_Data
	config = get_prototype_config()
	## 1. data prepare
	trainData = Post_Data(is_train = True, **config)
	testData = Post_Data(is_train = False, **config)
	### 2. model & train
	nn = Post_Prototype_RCNN_L2(assignment, **config)
	LR = config['LR']
	opt_  = torch.optim.SGD(nn.parameters(), lr=LR)
	best_auc = 0
	trainFeature_all, _ = trainData.get_all() 
	testFeature, testLabel = testData.get_all()
	for i in range(config['train_iter']):
		nn.generate_prototype(trainFeature_all)
		feat, label = trainData.next()
		loss, _ = nn(feat, label)
		opt_.zero_grad()
		loss.backward()
		opt_.step()
		loss_value = loss.data[0]
		if i % 1 == 0:
			score = post_evaluate(testFeature, testLabel, nn, **config)
			best_auc = score if score > best_auc else best_auc
			print('iter {}, auc:{} (best:{})'.format(i, str(score)[:5], str(best_auc)[:5]), end = ' ')
			if i > 0:
				print('{} sec'.format(str(time() - t1)[:4]))
			t1 = time()


def normalize_weight(weight_lst, upper, lower):
	## average
	weight0 = [i / sum(weight_lst) for i in weight_lst]
	f = lambda x:max(min(x,upper),lower)
	weight0 = list(map(f,weight0))
	return weight0

def train_weighted_post_prototype(assignment):
	from config import get_prototype_config
	from nn_model import Weighted_Post_Prototype_RCNN_L2
	from stream import Weighted_Post_Data
	config = get_prototype_config()
	every_iter = config['every_iter'] 
	## 1. data prepare
	trainData = Weighted_Post_Data(is_train = True, **config)
	testData = Weighted_Post_Data(is_train = False, **config)
	### 2. model & train
	nn = Weighted_Post_Prototype_RCNN_L2(assignment, **config)
	LR = config['LR']
	opt_  = torch.optim.SGD(nn.parameters(), lr=LR)
	best_auc = 0
	trainFeature_all, _, weight_all = trainData.get_all() 
	testFeature, testLabel, _ = testData.get_all()
	for i in range(config['train_iter']):
		nn.generate_prototype(trainFeature_all, weight_all)
		feat, label, weight = trainData.next()
		loss, _ = nn(feat, label, weight)
		opt_.zero_grad()
		loss.backward()
		opt_.step()
		#loss_value = loss.data[0]
		if i % 1 == 0:
			score = post_evaluate(testFeature, testLabel, nn, **config)
			best_auc = score if score > best_auc else best_auc
			if i % every_iter == every_iter - 1 and i > 0:
				print('iter {}, auc:{} (best:{})'.format(i, str(score)[:5], str(best_auc)[:5]), end = ' \n')
			#	print('{} sec'.format(str(time() - t1)[:4]))
			#t1 = time()
	
	trainData.restart()
	reweight = []
	for i in range(trainData.batch_number):
		feat, label, weight = trainData.next()
		reweight.extend(nn.measure_similarity(feat))
	reweight = normalize_weight(reweight, config['upper'], config['lower'])
	return reweight






if __name__ == '__main__':
	### finished
	#train_RCNN()
	### save data (1) data/heart_failure/train.npy  (2) data/heart_failure/test.npy
	from decision_tree import DT_learning
	reweight = None
	for i in range(5):
		print(' Epoch {}:'.format(i))
		assignment = DT_learning(reweight)
		#train_post_prototype(assignment)
		reweight = train_weighted_post_prototype(assignment)


	#train_Prototype_RCNN(assignment)
	#train_weighted_Prototype_RCNN(assignment)

	pass


