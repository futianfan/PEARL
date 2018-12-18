from sklearn import tree
import numpy as np 
from sklearn.metrics import roc_auc_score
from os import system
#from graphviz import Source
from time import time
from functools import reduce

from config import get_RCNN_config
from stream import Decision_Tree_Data

def evaluate(clf, Data, Label):
	prediction = clf.predict_proba(Data)
	prediction = prediction[:,1]
	prediction = list(prediction)
	label = list(Label)
	return str(roc_auc_score(label, prediction))[:5]


def DT_learning(weight_lst = None):
	config = get_RCNN_config()
	DT_train_data = Decision_Tree_Data(weight_lst = weight_lst, is_train = True, **config)
	DT_test_data = Decision_Tree_Data(weight_lst = None, is_train = False, **config)
	clf = tree.DecisionTreeClassifier(max_depth = 5)
	clf = clf.fit(DT_train_data.mat, np.array(DT_train_data.label), sample_weight = DT_train_data.weight_lst)

	auc = evaluate(clf, DT_test_data.mat, np.array(DT_test_data.label))
	##print('decision tree: test auc is {}'.format(auc))

	leaf_node_assign = clf.apply(DT_train_data.mat)
	leaf_node_assign = list(leaf_node_assign)
	leaf_node_set = set(leaf_node_assign)
	leaf_node_lst = list(leaf_node_set)
	leaf_node_lst.sort()
	leaf_node_assign = [list(filter(lambda i:leaf_node_assign[i] == leaf_node_indx , list(range(len(leaf_node_assign))))) for leaf_node_indx in leaf_node_lst]
	'''
	a = list(reduce(lambda x,y:x+y, leaf_node_assign))
	a.sort()
	assert a == list(range(len(a)))
	'''
	#print(leaf_node_assign)
	return leaf_node_assign


if __name__ == '__main__':
	assign = DT_learning()
	print(len(assign))



