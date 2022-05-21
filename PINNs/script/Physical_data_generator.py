# -*- coding: utf-8 -*-
"""
Created on Fri May  6 14:41:08 2022

@author: WeijieXia
"""

import pandas as pd
from optimizer import *
import numpy as np
from scipy.optimize import minimize

all_data = pd.read_csv((r'../data/all_data.csv'))
all_data = all_data.drop(columns=['Unnamed: 0'])

"battery"
SOC_0 = 0.1
C_ref = 4000
ch_rate = 1000 # W

"PV "
area=2.5764
panel_efficiency=0.203
loss=0.8
number_panels=40
all_data['Irradiation_W'] = all_data['Irradiation_W']*area*panel_efficiency*number_panels*loss
all_data.rename(columns={'Irradiation_W':'PV_generation','CONSUMPTION':'Consumption'},inplace=True)
all_data['Price'] = all_data['Price']/1000

'price data'
cost_pv = all_data.Price.sum()/10/len(all_data.Price)

"Store data"
acts = pd.DataFrame([])
SOCs = pd.DataFrame([])
SOC =  0.3000
input_d = pd.DataFrame([])
for i in range(len(all_data)-5):
# for i in range(10):
    # if i == 696:
        # pass
    # if i != 696:
        input_gprice = all_data.Price.iloc[i:i+5]
        input_pv = all_data.PV_generation.iloc[i:i+5]
        input_g = all_data.Consumption.iloc[i:i+5]
        X = physcial_optimizer(C_ref,ch_rate,SOC
                               ,input_pv,input_g
                               ,cost_pv,input_gprice)
        _act1 = X[0]
        SOC = SOC-X[0]/C_ref
        _act1 = pd.DataFrame([_act1])
        acts = pd.concat([acts,_act1],axis=0)
        _SOC = pd.DataFrame([SOC])
        SOCs = pd.concat([SOCs,_SOC],axis=0)
        z = pd.concat([input_gprice,input_pv,input_g],axis=0)
        z = np.array(z)
        z = pd.DataFrame([z])
        input_d = pd.concat([input_d,z],axis=0)
        print(i)
        
acts.rename(columns={0:'charging'},inplace=True)
SOCs.rename(columns={0:'SOC'},inplace=True)
acts.index = all_data.index[:len(acts)]
SOCs.index = all_data.index[:len(SOCs)]

#%%
output_data = pd.concat([all_data,acts,SOCs],axis=1)
output_data.to_csv('../data/learning_data.csv')

input_d.index = acts.index
output_data2 = pd.concat([input_d,acts,SOCs],axis=1)
output_data2.to_csv('../data/learning_data_2.csv')


