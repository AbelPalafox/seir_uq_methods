# -*- coding: utf-8 -*-
import numpy as np
from pandas import read_csv

def load_data(fname) :

    # reading data into a dataframe
    df_input = read_csv(fname)

    dates = np.asarray(df_input['Date'])
    data= df_input['Smoothed_Daily_Cases']
    
    times = np.arange(0,len(dates))
    
    return times, np.asarray(data), dates


def cast_value(value: str):
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value
    
def parser(fname) :
    
    params = {}
    with open(fname,'r') as fin :
        buffer = fin.readlines()
        for line in buffer :
            # add here a bad-format validation for cases without =
            key, val = line.split('=')             
            params[key.strip()] = cast_value(val.strip())
            
    return params

class Normalizer :
    
    def __init__(self, min_val, max_val) :
        
        self.min = min_val
        self.max = max_val
        
    def normalize(self, x) :
        return (x - self.min) / (self.max - self.min + 1e-8)  # 1e-8 para evitar divisiones por cero
        
    def denormalize(self, x) :
        return x * (self.max - self.min) + self.min