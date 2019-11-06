'''
utility functions and class
'''

import os
import pickle
from torch.utils.data import Dataset
import torch

def load_pickle(path):
    'load pickle files and return a dic-like class contain all pkl contents'
    pkls = os.listdir(path)
    assert pkls, 'no files exist!'

    class Mapper(dict):
        def __setitem__(self, key, item):
            self.__dict__[key] = item    
        def __getitem__(self, key):
            return self.__dict__[key]        
        def keys(self):
            return self.__dict__.keys()    
        def __len__(self):
            return len(self.__dict__)
        
    res = Mapper()
    for pkl in pkls:
        key = pkl.split(".")[0]
        pkl = os.path.join(path,pkl)
        with open(pkl,'rb') as f:
            res[key] = pickle.load(f)
    return res

class MyDataset(Dataset):
    def __init__(self,seq_dic):
        self.uids = list(seq_dic.keys())
        self.seqs = list(seq_dic.values())

    def __getitem__(self,x):
        return torch.tensor(self.seqs[x])

    def __len__(self):
        return len(self.seqs)

    
def my_collate(sample):
    train_seq = torch.tensor([i[0] for i in sample])
    test_seq = torch.tensor([i[1] for i in sample])
    return train_seq,test_seq

def get_dataset():
	'return dataset'
	with open('pkls/train_dic.pkl','rb') as f:
		train_dic = pickle.load(f)
	with open('pkls/test_dic.pkl','rb') as f:
		test_dic = pickle.load(f)
	return MyDataset(train_dic),MyDataset(test_dic)

def top_N_item(tensor,N=3):
    'get Top N result along with last dimension'
    assert len(tensor.shape)==2,\
           'tensor should be shape [batch,items]'
    res = []
    for t in tensor:
        res.append(torch.argsort(t,descending=True)[0:N])
    return torch.stack(res)

def MAP(predict,true_val):
    'calculate MAP@n'
    assert predict.shape==true_val.shape,\
           'shape should be the same'
    n = predict.shape[-1]
    res = (predict==true_val).float() # 1:true, 0:false
    pre_n = res.cumsum(-1)/torch.arange(1,n+1).float().to(res.device)
    map_n = (pre_n*res).mean(-1)
    return map_n

'''
a = torch.randn((10,6))
a = top_N_item(a)
b = torch.randint(0,6,(10,3))
MAP(a,b)
'''