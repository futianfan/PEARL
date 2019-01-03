from sklearn import tree
import numpy as np 
from sklearn.metrics import roc_auc_score
import os
from os import system
from graphviz import Source
from time import time
from functools import reduce


from config import get_RCNN_config
from stream import Decision_Tree_Data


wordmapfile = 'data/heart_failure/SNOW_vocabMAP.txt'
with open(wordmapfile, 'r') as fin:
	lines = fin.readlines()
	lines = [line.rstrip().split('\t')[1] for line in lines]
	feature_names = lines 

feature_names = [i.capitalize() for i in feature_names]
feature_names.extend([' '] * (1867 - len(feature_names)) )
dataFolder = 'result'
class_name = np.unique(['failure', 'success'])
dotfile = os.path.join(dataFolder, 'tree.dot')
dotfile2 = os.path.join(dataFolder, 'tree2.dot')
picfile = os.path.join(dataFolder, 'tree.png')



def change_dotfile(lines, filename):
	fout = open(filename, 'w')
	for i,line in enumerate(lines):
		if i <= 1 or i == len(lines) - 1:
			pass
		else:
			if "->" in line:
				fatherNode = int(line.split()[0])
				sonNode = int(line.split()[2])
				if sonNode - fatherNode == 1:
					line = str(fatherNode) + ' -> ' + str(sonNode) \
						+ ' [headlabel="no exist"] ;\n' 
				else:
					line = str(fatherNode) + ' -> ' + str(sonNode) \
						+ ' [headlabel="exist"] ;\n'
			else:
				
				stt = line.index("samples = ")
				#stt = stt - 2 if line[stt-2:stt] == '\\n' else stt
				endn = line.index("value = ")
				x = endn + 9										
				endn = line[endn:].index(']') + endn	
				num_str = line[x:endn]
				#print(num_str)
				num1, num2 = num_str.split(',')		
				num1, num2 = float(num1), float(num2)
				prob = num2 / (num1 + num2)
				endn = endn + 2 if line[endn+1:endn+3] == '\\n' else endn
				line = line[:stt] + line[endn+1:]
				pass
		if "<= 0.5" in line: 
			line = line.replace('<= 0.5', '')
		if 'class =' in line:
			if 'success' in line:
				#line = line.replace('class = success', 'prediction: success')
				line = line.replace('class = success', 'class = success, failure prob: '+str(prob * 100)[:4] + '\%')				
			elif 'failure' in line: 
				line = line.replace('class = failure', 'class = failure, failure prob: '+str(prob * 100)[:4] + '\%')
		fout.write(line) 
	fout.close()





def DT_learning(weight_lst = None):
	config = get_RCNN_config()
	DT_train_data = Decision_Tree_Data(weight_lst = weight_lst, is_train = True, **config)
	DT_test_data = Decision_Tree_Data(weight_lst = None, is_train = False, **config)
	clf = tree.DecisionTreeClassifier(max_depth = 5)
	clf = clf.fit(DT_train_data.mat, np.array(DT_train_data.label), sample_weight = DT_train_data.weight_lst)

	#tree.export_graphviz(clf, out_file = dotfile, feature_names = feature_name[1:], impurity = False,  class_names = class_name, proportion = False)
	tree.export_graphviz(clf, out_file = dotfile, 
							  impurity = False, \
							  feature_names = feature_names, \
							  class_names = class_name, \
							  proportion = False
						)
	with open(dotfile, 'r') as fin:
		lines = fin.readlines()
	change_dotfile(lines, dotfile2)
	system("dot -Tpng " + dotfile2 + " -o " + picfile)

if __name__ == "__main__":
	DT_learning()






