import tensorflow as tf



def train_rcnn():
	from config import get_RCNN_config
	from stream import TF_Sequence_Data, _embed_file_2_numpy_embed_mat 
	from model_tf import Rcnn_Base
	config = get_RCNN_config()
	embedding = _embed_file_2_numpy_embed_mat(config['embed_file'], config['admis_dim'])
	trainData = TF_Sequence_Data(is_train = True, **config)
	rcnn_base = Rcnn_Base(**config)

	for i in range(config['train_iter']):
		seq_embed, seq_len, label = trainData.next(embedding)
		loss = rcnn_base.train(seq_embed, label, seq_len)
		print(loss)




if __name__ == '__main__':
	train_rcnn()





