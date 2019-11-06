'''
split train/test set
'''
import pickle
import random
import os

SEED = sum(ord(char) for char in 'unemployed') #1090
with open('pkls/uid_seq.pkl','rb') as f:
    uid_seq = pickle.load(f)
N = len(uid_seq)
TEST_RATE = 0.08
PKL_PATH = 'pkls'

def split_():	
	test_n = int(N*TEST_RATE)
	random.seed(SEED)
	test_sel = set(random.sample(range(N),test_n))
	train_dic = dict()
	test_dic = dict()

	for no,(k,v) in enumerate(uid_seq.items()):
	    if no in test_sel:
	        test_dic[k] = v
	    else:
	        train_dic[k] = v

	return train_dic,test_dic

def save_pkl(train_dic,test_dic):
	path = os.path.join(PKL_PATH,'train_dic.pkl')
	with open(path,'wb') as f:
		pickle.dump(train_dic, f)
	
	path = os.path.join(PKL_PATH,'test_dic.pkl')
	with open(path,'wb') as f:
		pickle.dump(test_dic, f)

def main():
	train_dic,test_dic = split_()
	save_pkl(train_dic, test_dic)

if __name__ == '__main__':
	main()
