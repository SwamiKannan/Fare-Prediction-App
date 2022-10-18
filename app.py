import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as slt
from utils import preprocessing
from model import FarePredictor

torch.manual_seed(33)

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
    with torch.no_grad():
        value=fare_predictor.forward(cat_cols,cont_cols).item()
        slt.success('The predicted fare is: USD '+str(round(value,2))+'   :thumbsup:')
    
slt.button('PREDICT MY FARE !', on_click=predict)
