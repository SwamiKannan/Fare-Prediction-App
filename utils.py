import numpy as np
import pandas as pd
import torch

def haversine_distance(df, lat1, long1, lat2, long2):
    """
    Calculates the haversine distance between 2 sets of GPS coordinates in df
    """
    r = 6371  # average radius of Earth in kilometers
       
    phi1 = np.radians(df[lat1])
    phi2 = np.radians(df[lat2])
    
    delta_phi = np.radians(df[lat2]-df[lat1])
    delta_lambda = np.radians(df[long2]-df[long1])
     
    a = np.sin(delta_phi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = (r * c) # in kilometers
    return d

def preprocessing(pickup_datetime, pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,passenger_count):
    dict_1={'samp_pickup_datetime':pickup_datetime,'samp_pickup_longitude':pickup_longitude,'samp_pickup_latitude':pickup_latitude,
       'samp_dropoff_longitude':dropoff_longitude,'samp_dropoff_latitude':dropoff_latitude,'samp_passenger_count':passenger_count
      }
    df=pd.DataFrame(dict_1,columns=dict_1.keys(),index=[0])
    df['samp_pickup_datetime']=pd.to_datetime(df['samp_pickup_datetime'])
    df['distance']=haversine_distance(df,'samp_pickup_latitude', 'samp_pickup_longitude', 'samp_dropoff_latitude', 'samp_dropoff_longitude')
    df['hours']=df['samp_pickup_datetime'].dt.hour
    df['AM_PM']=np.where(df['hours']<12,"AM","PM")
    df['weekday']=df['samp_pickup_datetime'].dt.strftime('%a')
    cats=['hours','weekday','AM_PM']
    for cat in cats:
        df[cat]=df[cat].astype('category')
    cat_cols=torch.tensor(np.stack([df[cat].cat.codes for cat in cats],axis=1),dtype=torch.long)
    conts=['samp_pickup_longitude','samp_pickup_latitude', 'samp_dropoff_longitude', 'samp_dropoff_latitude', 'samp_passenger_count', 'distance']
    cont_cols=torch.tensor(np.stack([df[col].values for col in conts],axis=1),dtype=torch.float)
    return cat_cols, cont_cols
