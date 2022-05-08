# -*- coding: utf-8 -*-
"""
Created on Fri May  6 14:27:59 2022

@author: WeijieXia
"""
import pandas as pd
import datetime as dt

'read data'
con_data = pd.read_csv(r'../data/consumption_data0.csv')
irr_data = pd.read_csv(r'../data/irradiation_file.csv')
price_data = pd.read_csv(r'../data/Day_ahead_1hour_2019_2022 (3).csv',parse_dates = ['Delivery_Start'])

'process price data'
price_data = price_data[(price_data.Delivery_Start>= dt.datetime(2021, 1, 1)) & 
                        (price_data.Delivery_Start< dt.datetime(2022, 1, 1))]
price_data = price_data['Price']
price_data.index = irr_data.index

'concat data'
all_data = pd.concat([irr_data['Irradiation_W'],price_data,con_data],axis=1) 
# all_data.to_csv(r'../data/all_data.csv')