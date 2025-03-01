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