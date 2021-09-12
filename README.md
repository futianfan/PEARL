# PEARL


## conda 

```bash
conda create -n pearl python=3.7 
conda activate pearl 
pip install torch 
pip install scikit-learn 
pip install matplotlib 
```

## data 

`data/`: data folder

`src/stream.py`: process data


## config 

`src/config.py`: configuration 


## model 

`src/decision_tree.py`: decision-tree for interpretable rule

`src/model_torch.py`: NN model (Pytorch) 

## train 

```bash
python src/train_torch.py
```



## Contact
Tianfan Fu (futianfan@gmail.com)
