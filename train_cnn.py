'''train cnn model'''

from collections import Counter
from itertools import chain
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from gensim.models import Word2Vec
from utli import *
from models import *
import numpy as np
######## model parameters ########
NUM_EPOCH=6
EMB_DIM = 100
CONV_PARAMS = [
    (8,(1,EMB_DIM),0,1),
    (8,(2,EMB_DIM),0,1),
    (8,(3,EMB_DIM),0,1),
    (8,(5,EMB_DIM),0,1),
    (8,(2,EMB_DIM),0,2),
    (8,(3,EMB_DIM),0,2),
]
LINEAR_PARAMS = [
    (48,512),(512,512)
]
DROPOUT = 0.1
FREEZE_EMB=False
MIN_COUNT = 5
NUM_NS = 10
LR = 0.001
BATCH_SIZE = 64
W2V_PATH = 'word2vec_ns5_win5.model'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
##################################

def get_word2vec_emb(path,num_items):
	'get the embedding layer from pretrain'
	model = Word2Vec.load(path)
	weights = torch.empty((num_items,EMB_DIM)).normal_(mean=0,std=0.01)
	for v in model.wv.vocab:
	    weights[int(v)] = torch.from_numpy(model.wv.word_vec(v))
	embs = nn.Sequential(
		nn.Embedding.from_pretrained(weights,freeze=FREEZE_EMB,padding_idx=0),
		nn.Dropout(DROPOUT)
	)	
	return embs

def filter_count(data:dict,MIN_COUNT)->dict:
	'transform item_id to 0 if item count< MIN_COUNT'
	item_counter = Counter(chain(*data.values()))
	valid_set = set(k for k,v in item_counter.items() if v>(MIN_COUNT-1))
	data_cleaned = data.copy()
	def clean(x):
	    res = x.copy()
	    for i in range(10):
	        if res[i] not in valid_set:
	            res[i]=0
	    return res
	return { k:clean(v) for k,v in data_cleaned.items()}

def get_num_items(train_dic,test_dic)->int:
	'get the num of items'
	train_set = set(chain(*train_dic.values()))
	test_set = set(chain(*test_dic.values()))
	#+1 since index start from 1 for padding embedding
	return max(train_set.union(test_set))+1 

def prepare_data():
	pkls = load_pickle('pkls')
	if MIN_COUNT:
		train_dic = filter_count(pkls.train_dic,MIN_COUNT)
	else:
		train_dic = pkls.train_dic
	test_dic = pkls.test_dic
	return train_dic,test_dic

def prepare_model(train_dic,num_items,use_pretrain=False):
	train_dataset = MyDataset(train_dic)
	loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)	
	if use_pretrain:
		embs = get_word2vec_emb(W2V_PATH, num_items)		
	else:
		embs = nn.Sequential(nn.Embedding(num_items,EMB_DIM,0),nn.Dropout(DROPOUT))

	cnn = CNN(CONV_PARAMS,LINEAR_PARAMS,num_items,embs).to(DEVICE)
	bpr = BPRLoss().to(DEVICE)
	ns = NegativeSamplingLoss(NUM_NS).to(DEVICE)
	optimizer = torch.optim.Adam(cnn.parameters(),lr=LR)	
	return loader,cnn,bpr,ns,optimizer

def train_model(loader,cnn,bpr,ns,optimizer):
	for epoch in range(NUM_EPOCH):
	    t0 = time.time()
	    loss = 0
	    for i in loader:
	        x = i[:,0:10].unsqueeze(1).to(DEVICE)
	        y = i[:,10].to(DEVICE)        
	        out = cnn(x,y)
	        #out = bpr(out)
	        out = ns(out,y)
	        loss +=out.item()
	        cnn.zero_grad()
	        out.backward()
	        optimizer.step()
	    print ('Epoch [{}/{}], Loss: {:.4f}, Time: {} sec'
	           .format(epoch+1, NUM_EPOCH, loss, int(time.time()-t0))
	          )
	    #calculate MAP,prec@1 for train_set every 5 epoch
	    if (epoch+1)%5==0:
	        with torch.no_grad():
	            maps = []
	            prec1 = []
	            for i in loader:
	                x = i[:,0:10].unsqueeze(1).to(DEVICE)
	                y = i[:,10].to(DEVICE)        
	                true_y = i[:,10:].to(DEVICE)        
	                out = cnn(x)
	                pred = top_N_item(out)
	                maps.append(MAP(pred,true_y).tolist())
	                prec1.append((pred[:,0]==y).tolist())
	        print("\t\tMAP: {}".format(np.mean(list(chain(*maps)))))
	        print("\t\tprec@1: {}".format(np.mean(list(chain(*prec1)))))
	return cnn,bpr,ns,optimizer	

def main():
	train_dic,test_dic = prepare_data()
	num_items = get_num_items(train_dic, test_dic)
	loader,cnn,bpr,ns,optimizer  = prepare_model(train_dic,num_items)
	cnn,_,_,_,_ = train_model(loader,cnn,bpr,ns,optimizer)


if __name__ == '__main__':
	main()


