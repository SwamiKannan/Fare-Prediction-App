import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as slt
from utils import preprocessing

class FarePredictor(nn.Module):
    def __init__(self,cat_cols,cont_cols,layer_count,output_features,embed_size_list,p=0.5): 
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

slt.title('Fare Prediction :taxi:')
pickup_time=slt.text_input('Enter the pickup time', '2010-04-15 16:00:00')
pickup_longitude=slt.number_input('Enter the longitude of pickup',-180.0,180.0)
pickup_latitude=slt.number_input('Enter the latitude of pickup',-180.0,180.0)
dropoff_longitude=slt.number_input('Enter the longitude of dropoff',-180.0,180.0)
dropoff_latitude=slt.number_input('Enter the latitude of dropoff',-180.0,180.0)
passenger_count=slt.number_input('Enter the number of passengers',1,4)

cat_col=[1,3]
cont_col=[1,6]
fare_predictor=FarePredictor(torch.zeros((1,3)),torch.zeros((1,6)),layer_count=[200,100],output_features=1,embed_size_list=[(24, 12), (7, 4), (2, 1)],p=0.4)
fare_predictor.load_state_dict(torch.load('uber_model_weights.pt'))
fare_predictor.eval()

def predict():
    cat_cols, cont_cols = preprocessing(pickup_time,pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,passenger_count)
    #fare_predictor=FarePredictor(cat_cols,cont_cols,layer_count=[200,100],output_features=1,embed_size_list=[(24, 12), (7, 4), (2, 1)],p=0.4)
    # fare_predictor.load_state_dict(torch.load('uber_model_weights.pt'))
    # fare_predictor.eval()
    with torch.no_grad():
        value=fare_predictor.forward(cat_cols,cont_cols).item()
        slt.success('The predicted fare is: USD '+str(round(value,2))+'   :thumbsup:')
    
slt.button('PREDICT MY FARE !', on_click=predict)
