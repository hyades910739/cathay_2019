'''
Implement CNN-based next-item prediction model
reference:
	* CNN model : (2014) Convolutional Neural Networks for Sentence Classification
	* BPR loss : (2017) Recurrent Neural Networks with Top-k Gains for Session-based Recommendations
	* NS loss :  (2018) Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding.        
'''
import torch
import torch.nn as nn
from utli import *

class Conv_Pool(nn.Module):
    '''
     A conv2d layer + maxpooling layer
    '''
    def __init__(self,out,kernel,pad,dila):
        super(Conv_Pool,self).__init__()
        self.pool_size = 10-dila*(kernel[0]-1)+2*pad
        assert self.pool_size>0,\
               'Given conv2d parameters are invalid:{}'.format(
                    (out,kernel,pad,dila)
               )
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=out,
            kernel_size= kernel,
            padding=pad,
            dilation=(dila,1)
        )
        self.pooling = nn.MaxPool1d(self.pool_size)
   
    def forward(self,x):
        x = self.conv(x).squeeze()
        x = self.pooling(x).squeeze()
        return x


class CNN(nn.Module):
    '''
    A CNN-based recommendation model.
    
    params:
    ----
    conv_params: list of tuples contains arguments for Conv_Pool
    linear_params: list of tuples contains arguments for nn.Linear layer (in and out features)
    num_items: number of output items
    embs: a nn.Embedding layer or nn.Sequential layer that eats longtensor as input
    bn: Bool, whether to use batchNorm in linear layer
    '''
    def __init__(self,
                 conv_params,
                 linear_params,
                 num_items,
                 embs,
                 bn=True):
        #check dimension
        conv_output = sum(i[0] for i in conv_params)
        assert conv_output==linear_params[0][0],\
               'dimension is invalid from conv to linear layer'
        super(CNN,self).__init__()
        self.embs = embs
        self.conv_params = conv_params
        self.linear_params = linear_params
        self.convs = nn.ModuleList(Conv_Pool(*p) for p in self.conv_params)
        if bn: 
            getter = lambda a,b: nn.Sequential(nn.BatchNorm1d(a),nn.Linear(a,b),nn.ReLU())
        else:
            getter = lambda a,b: nn.Sequential(nn.Linear(a,b),nn.ReLU())
        
        self.linear = nn.Sequential(*(getter(a,b) for a,b in self.linear_params))
        self.out_w = nn.Parameter(
            torch.empty((self.linear_params[-1][-1],num_items)).normal_(mean=0,std=0.01),
            
        )
        self.out_b = nn.Parameter(
            torch.empty((num_items)).normal_(mean=0,std=0.01),            
        )        
        self.softmax = nn.LogSoftmax(-1)
        
    def forward(self,x,sel_out=None):
        x = self.embs(x)
        x = torch.cat([conv(x) for conv in self.convs],-1)
        x = self.linear(x)
        if sel_out is not None:
            x = torch.matmul(x,self.out_w[:,sel_out]) + self.out_b[sel_out]
        else:
            x = torch.matmul(x,self.out_w) + self.out_b           
        return x                                 


class BPRLoss(nn.Module):
    '''
        Implement Bayesian Personalized Ranking Loss
        input
        -----
        x: tensor with shape (batch_size,batch_size), 
           where the first dim is batch, last dim is item score.
           The target item should locate in diag.
    
    '''
    def __init__(self):
        super(BPRLoss,self).__init__()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        'note that the target item is the diag element'
        n = x.shape[0]
        diags = x.diag().unsqueeze(-1).expand((n,n))
        x = diags-x
        # zero grad for "target miuns target" situation.
        # for example, every diag element of x
        x = x*(x!=0).float() 
        x = self.sigmoid(x)
        x = torch.log(x)
        return -x.mean()                   


class NegativeSamplingLoss(nn.Module):
    '''
        Implement Negative Sampling Loss
        Note that the items with top n high score will select as  negative samples.
        NS can also sample at random, not implement yet.
        ----
        param:
        n: int, number of negative samples        
    '''
    def __init__(self,n):
        super(NegativeSamplingLoss,self).__init__()
        self.n = n
        
    def forward(self,x,sel_out):        
        mask = self._get_mask(sel_out)
        pos = -torch.mean(torch.log(torch.sigmoid(x.diag())))
        indices = top_N_item(x*mask,N=self.n)
        # select the indices items as neg sample
        neg = x[torch.arange(indices.shape[0]),indices.t()].reshape((-1,))
        neg = -torch.mean(torch.log(1 - torch.sigmoid(neg)))
        loss = pos+neg
        return loss
    
    def _get_mask(self,sel_out):
        '''
           Given output matrix x with shape (n_batch,item),
           for user in row i, the target item is ith col. i.e, the diag element of matrix x is the target items.
           For any user, to prevent non-diag element equals target element
           and choosed as negative sample, we must mask it as 0
           
           example:
           >> cls = NegativeSamplingLoss(3)
           >> cls._get_mask(torch.tensor([1,2,3,2,1]))
        '''        
        n = sel_out.shape[0]
        x = sel_out.unsqueeze(0).expand((n,n))
        y = sel_out.unsqueeze(-1).expand((n,n))
        mask = (x!=y).float()
        return mask
        
        
    


if __name__ == '__main__':
	conv_params = [
	    (5,(1,100),0,1),
	    (5,(2,100),0,1),
	    (5,(3,100),0,2),
	    (5,(5,100),0,2),  
	]
	linear_params = [
	    (20,100),(100,100)
	]
	num_items = 50
	embs = nn.Sequential(nn.Embedding(50,100),nn.Dropout(0.1))

	cnn = CNN(conv_params,linear_params,num_items,embs)
	x = torch.randint(0,48,(3,1,10))
	print(cnn)
	print(cnn(x))