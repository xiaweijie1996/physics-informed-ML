# -*- coding: utf-8 -*-
"""
Created on Sat May  7 11:44:39 2022

@author: WeijieXia
"""

import pandas as pd
import numpy as np 

output_data = pd.read_csv(r'../data/learning_data.csv')

PV_generation = output_data.PV_generation.iloc[0:24*300]
PV_generation = np.array(PV_generation)
PV_generation = PV_generation.reshape(300,24)
PV_generation_ave = sum(PV_generation)/300

Price = output_data.Price.iloc[0:24*300]
Price = np.array(Price)
Price = Price.reshape(300,24)
Price_ave = sum(Price)/300

Consumption = output_data.Consumption.iloc[0:24*300]
Consumption = np.array(Consumption)
Consumption = Consumption.reshape(300,24)
Consumption_ave = sum(Consumption)/300

charging = output_data.charging .iloc[0:24*300]
charging = np.array(charging)
charging = charging.reshape(300,24)
charging_ave = sum(charging)/300

SOC = output_data.SOC.iloc[0:24*300]
SOC = np.array(SOC)
SOC = SOC.reshape(300,24)
SOC_ave = sum(SOC)/300
