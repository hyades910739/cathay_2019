'''train a word2vec embedding'''
from utli import *
from gensim.models import Word2Vec

pkls = load_pickle('pkls')
lines = [v for v in pkls.train_dic.values()]
lines = [ [str(i) for i in li] for li in lines]
model = Word2Vec(lines, size=100, window=5, min_count=5, workers=4,sg=1,iter=20)
model.save('word2vec_ns5_win5.model')