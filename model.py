import torch.nn as nn
import torch.nn.functional as F
import torch

class FarePredictor(nn.Module):
    def __init__(self,cat_cols,cont_cols,layer_count,output_features,embed_size_list,p=0.5): 
        '''
        args:
        cat_cols is the tensor of all categorical values (pre-embedding)
        n_cont=number of continuous variables (for batch normalization)
        cont_cols is the tensor of all continuous values
        input_features - number of parameters of input
        layer_count - a tuple of number of nodes of each hidden layer
        output_features = number of outputs expected
        embed_size_list is list of embedding sizes for the categorical values
        p = basically, the % of nodes to be nullified during dropout layer
       
       Approach: In the constructor, create all the layers (Linear, ReLU, Batch and Dropout) for each hidden layer as per layer_count)
       and add them to sequential(). Fwd() will have all the data manipulation and final embedding
       
        '''
        super().__init__()
        self.output_features=output_features
        self.layer_count=layer_count
        self.embed_size_list=embed_size_list
        self.embeddings=[nn.Embedding(base_dim, target_dim) for base_dim, target_dim in embed_size_list]
        n_cont=cont_cols.shape[1]
        self.batch_norm=nn.BatchNorm1d(n_cont)
        self.dropout=nn.Dropout(p=0.5)
        n_cont=cont_cols.shape[1]
        n_in=sum(nf for ni,nf in self.embed_size_list)+n_cont
        self.layers=[]
        for l in self.layer_count:
            self.layers.append(nn.Linear(n_in,l))
            self.layers.append(nn.ReLU(inplace=True))
            self.layers.append(nn.Dropout(p))
            self.layers.append(nn.BatchNorm1d(l))            
            n_in=l
        self.layers.append(nn.Linear(self.layer_count[-1],self.output_features))
        self.final_layers=nn.Sequential(*self.layers)
        
    def forward(self,cat_cols, cont_cols):
        embeds=[]
        for i,e in enumerate(self.embeddings):
            embeds.append(e(cat_cols[:,i]))
        cat_final=torch.cat(embeds,axis=1)
        cont_cols=self.batch_norm(cont_cols)
        cat_final=self.dropout(cat_final)
        self.X=torch.cat((cat_final, cont_cols),axis=1)
        self.X=self.final_layers(self.X)
        return self.X