import os 

def get_RCNN_config():
	config = {}

	### data-processing
	config['HF_folder'] = './data/heart_failure' 
	config['train_file'] = os.path.join(config['HF_folder'], 'training_data_1.txt')
	config['test_file'] = os.path.join(config['HF_folder'], 'test_data_1.txt') 
	config['embed_file'] = os.path.join(config['HF_folder'], 'training_model_by_word2vec_1.vector')
	# assert os.path.exists(config['train_file'])

	config['new_train_file'] = os.path.join(config['HF_folder'], 'train.npy')
	config['new_test_file'] = os.path.join(config['HF_folder'], 'test.npy')

	config['embed_dim'] = 100
	config['admis_dim'] = 1867
	config['max_length'] = 30
	config['num_class'] = 2
	config['batch_size'] = 32  ### 32, 256 
	config['big_batch_size'] = 256
	config['train_iter'] = 30000 ## int(2e3)

	### Neural Network
	config['cnn_kernel_size'] = 10
	config['cnn_stride'] = 1 
	config['cnn_out_channel'] = 150
	config['maxpool_size'] = 3
	config['maxpool_stride'] = 1
	config['rnn_hidden_size'] = 50 
	config['rnn_num_layer'] = 1
	config['batch_first']  = True
	config['bidirectional'] = True

	config['LR'] = 1e-1		### 1e-1

	### Decision Tree
	### reweight
	return config


def get_TF_RCNN_config():
	config = {}

	### data-processing
	config['HF_folder'] = './data/heart_failure' 
	config['train_file'] = os.path.join(config['HF_folder'], 'training_data_1.txt')
	config['test_file'] = os.path.join(config['HF_folder'], 'test_data_1.txt') 
	config['embed_file'] = os.path.join(config['HF_folder'], 'training_model_by_word2vec_1.vector')
	# assert os.path.exists(config['train_file'])

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

	### Decision Tree
	### reweight
	return config

def get_TF_weight_prototype_RCNN_config():
	config = {}

	### data-processing
	config['HF_folder'] = './data/heart_failure' 
	config['train_file'] = os.path.join(config['HF_folder'], 'training_data_1.txt')
	config['test_file'] = os.path.join(config['HF_folder'], 'test_data_1.txt') 
	config['embed_file'] = os.path.join(config['HF_folder'], 'training_model_by_word2vec_1.vector')
	assert os.path.exists(config['train_file'])

	######
	config['weight_file'] = ''
	config['eta'] = 1e-7   #### prototype loss 
	######


	config['new_train_file'] = os.path.join(config['HF_folder'], 'train.npy')
	config['new_test_file'] = os.path.join(config['HF_folder'], 'test.npy')

	config['embed_dim'] = 100
	config['admis_dim'] = 1867
	config['max_length'] = 50
	config['num_class'] = 2
	config['batch_size'] = 512  ### 8, 64, 512  
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

	### Decision Tree
	### reweight
	return config




def get_prototype_config():
	config = {}

	### data-processing
	config['HF_folder'] = './data/heart_failure' 
	config['train_file'] = os.path.join(config['HF_folder'], 'training_data_1.txt')
	config['test_file'] = os.path.join(config['HF_folder'], 'test_data_1.txt') 
	config['embed_file'] = os.path.join(config['HF_folder'], 'training_model_by_word2vec_1.vector')
	assert os.path.exists(config['train_file'])

	config['new_train_file'] = os.path.join(config['HF_folder'], 'train.npy')
	config['new_test_file'] = os.path.join(config['HF_folder'], 'test.npy')

	config['embed_dim'] = 100
	config['admis_dim'] = 1867
	config['max_length'] = 50
	config['num_class'] = 2
	config['batch_size'] = 256  ### 32, 256 
	config['big_batch_size'] = 256

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

	config['train_iter'] = 100 ## int(2e3)
	config['every_iter'] = 20
	config['upper'] = 3
	config['lower'] = 1.0 / 3


	### Decision Tree
	### reweight
	return config



if __name__ == '__main__':
	get_RCNN_config()








