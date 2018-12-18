PEARL
=============================

data/

|------	heart_failure/

|------	mimic3/: 


./EHR/interpretable_healthcare/mimicIII2/data



==============================


data + model

(0): config.py: 
	data,nn,dt,main,  for nn: train/dev/test

(1): stream.py:
	nn,dt, initial_weight=1.0

(2): decision_tree.py: 
	decision-tree => interpretable rule
	input:
	output: rule + assignment
(3): nn_model.py, nn_train.py(train+test+evaluate), nn_run.py(including reweight) 
	input: assignment results based on decision tree.
	output: accuracy, weight 
(4): main.py
	python nn_train.py
	



