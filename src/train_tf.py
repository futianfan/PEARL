import tensorflow as tf
import numpy as np 

np.random.seed(4)
tf.set_random_seed(1)

def test(model, All_data):
	from sklearn.metrics import roc_auc_score
	batch_num = All_data.batch_number
	label_all = []
	predict_all = [] 
	for i in range(batch_num):
		next_data = All_data.next()
		data, data_len, label = next_data[0], next_data[1], next_data[2]
		output_prob = model.evaluate(data, data_len)
		output_prob = output_prob[0]
		output_prob = [i[1] for i in output_prob]
		label_all.extend(label)
		predict_all.extend(output_prob)
	return roc_auc_score(label_all, predict_all)

def train_rcnn():
	from config import get_TF_RCNN_config
	from stream import TF_Sequence_Data 
	from model_tf import Rcnn_Base
	config = get_TF_RCNN_config()
	trainData = TF_Sequence_Data(is_train = True, **config)
	TestData = TF_Sequence_Data(is_train = False, **config)
	rcnn_base = Rcnn_Base(**config)

	batch_num = trainData.batch_number
	total_loss = 0
	for i in range(config['train_iter']):
		seq_embed, seq_len, label = trainData.next()
		loss = rcnn_base.train(seq_embed, label, seq_len)
		total_loss += loss 
		if i > 0 and i % batch_num == 0:
			auc = test(rcnn_base, TestData)
			print('Loss: {}, test AUC {}.'.format(str(total_loss / batch_num)[:5], str(auc)[:5]))
			total_loss = 0 

def train_weight_rcnn():
	from config import get_TF_weight_prototype_RCNN_config
	from stream import TF_weighted_Sequence_Data 
	from model_tf import Weighted_Rcnn
	config = get_TF_weight_prototype_RCNN_config()
	trainData = TF_weighted_Sequence_Data(is_train = True, **config)
	TestData = TF_weighted_Sequence_Data(is_train = False, **config)
	rcnn_base = Weighted_Rcnn(**config)

	batch_num = trainData.batch_number
	total_loss = 0
	for i in range(config['train_iter']):
		seq_embed, seq_len, label, weight = trainData.next()
		loss = rcnn_base.train(seq_embed, label, seq_len, weight)
		total_loss += loss 
		if i > 0 and i % batch_num == 0:
			auc = test(rcnn_base, TestData)
			print('Loss: {}, test AUC {}.'.format(str(total_loss / batch_num)[:5], str(auc)[:5]))
			total_loss = 0 




if __name__ == '__main__':
	#train_rcnn()
	train_weight_rcnn()

	




