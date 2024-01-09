#import sys
#sys.path.append('E:/summer_intern/Hua_zheng_Wang/source_localization/DySL')
from torch.utils.data import Dataset, DataLoader
import pickle
class CustomDataset(Dataset):
    def __init__(self, pickle_file_path):
        self.data = self.load_data(pickle_file_path)

    def load_data(self, pickle_file_path):
        with open(pickle_file_path, 'rb') as f:
            data = pickle.load(f)
        #print(data)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # You can process the sample if needed
        return sample

#from dataloader import CustomDataset
import networkx as nx
import numpy as np
from pathlib import Path
import torch
import copy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import sys
sys.path.append('E:/summer_intern/Hua_zheng_Wang/source_localization/DySL/code/baselines/IVGD/main')
from main.i_deepis import i_DeepIS, DiffusionPropagate
from main.models.MLP import MLPTransform
from main.training import train_model, FeatureCons,get_predictions_new_seeds
from main.utils import load_dataset

#In this part, we train and test on single model from IVGD

pickle_file_path = "E:/summer_intern/Hua_zheng_Wang/source_localization/DySL/datasets/bitcoin-alpha/saved/saved_8.pkl"
custom_dataset = CustomDataset(pickle_file_path)
adjacency_matrix = nx.to_numpy_matrix(custom_dataset['multi_graph'])
#print(adjacency_matrix[0])
#prob_matrix=adjacency_matrix
#print(custom_dataset['tau'])
#print(prob_matrix.shape)
prob_matrix=[]
for row in adjacency_matrix:
    row=np.array(row).flatten()
    temp_ar=[]
    for element in row:
        #print(element)
        if element==1:
            temp_ar.append(float(custom_dataset['tau']))
        else:
            temp_ar.append(0.01)
    prob_matrix.append(temp_ar)
#prob_matrix=[[float(custom_dataset['tau']) if element == 1 else element for element in row] for row in prob_matrix]
dataset=np.array(custom_dataset['influ_list'])
graph=custom_dataset['multi_graph']

num_training= int(len(dataset)*0.8)
train_set=dataset[:num_training]
prob_matrix=np.array(prob_matrix)
train_set=np.array(train_set)
model_name = 'deepis' # 'deepis',
num_node=train_set.shape[2]
ndim = 5
propagate_model = DiffusionPropagate(prob_matrix, niter=2)
fea_constructor = FeatureCons(model_name, ndim=ndim)
fea_constructor.prob_matrix = prob_matrix
device = 'cpu'  # 'cpu', 'cuda'
#idx_split_args should be adapted to different datasets
args_dict = {
    'learning_rate': 1e-4,
    'λ': 0,
    'γ': 0,
    'ckpt_dir': Path('.'),
    'idx_split_args': {'ntraining': int(num_node/3), 'nstopping': int(num_node/3), 'nval': int(num_node/3), 'seed': 2413340114},
    'test': False,
    'device': device,
    'print_interval': 1,
    'batch_size': None,

}
if model_name == 'deepis':
    gnn_model = MLPTransform(input_dim=ndim, hiddenunits=[ndim, ndim], num_classes=1,device=device)
else:
    pass
model = i_DeepIS(gnn_model=gnn_model, propagate=propagate_model)
dataset_name='bitcoin'
for timestamp in range(train_set.shape[1]-1):
    print(timestamp)
    print(train_set.shape[1])
    model, result = train_model(model_name + '_' + dataset_name, model, fea_constructor, prob_matrix, train_set, **args_dict, timestamp=timestamp+1)
    #print("diffusion mae:"+str(me_op(influ_pred,influ_mat_list[0,:,1])))
    torch.save(model,"i-deepis_"+"bitcoin"+str(timestamp+1)+".pt")