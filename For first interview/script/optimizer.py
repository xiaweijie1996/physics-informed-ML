# -*- coding: utf-8 -*-
"""
Created on Thu May  5 18:20:40 2022

@author: WeijieXia
"""

from scipy.optimize import minimize
import numpy as np
import pandas as pd


def physcial_optimizer(C_ref,ch_rate,SOC
                       ,input_pv,input_g
                       ,cost_pv,input_gprice,
                       SOC_low=0.05):
    """
    SOC_0: intinal state of charge
    c_reg: capacity of the battery
    L_PV: input data including PV generation, load
    cost_p: marginalcost of PV
    cost_g: electriciry price
    """

    sum1 = np.sum(input_pv*cost_pv)
    sum2 = np.sum(input_g*input_gprice-input_pv*input_gprice)
    sum_t = float(sum1+sum2)
    fun = lambda x:sum_t-(input_gprice.iloc[0]*x[0]+input_gprice.iloc[1]*x[1]+input_gprice.iloc[2]*x[2]+
                          input_gprice.iloc[3]*x[3]+input_gprice.iloc[4]*x[4])
    
    cons = (
            {'type':'ineq','fun':lambda x:1+(x[0])/C_ref-SOC},
            {'type':'ineq','fun':lambda x:1+(x[0]+x[1])/C_ref-SOC},
            {'type':'ineq','fun':lambda x:1+(x[0]+x[1]+x[2])/C_ref-SOC},
            {'type':'ineq','fun':lambda x:1+(x[0]+x[1]+x[2]+x[3])/C_ref-SOC},
            {'type':'ineq','fun':lambda x:1+(x[0]+x[1]+x[2]+x[3]+x[4])/C_ref-SOC},
            
            {'type':'ineq','fun':lambda x:-(x[0]+x[1]+x[2]+x[3]+x[4])/C_ref+SOC-SOC_low},
            {'type':'ineq','fun':lambda x:-(x[0]+x[1]+x[2]+x[3])/C_ref+SOC-SOC_low},
            {'type':'ineq','fun':lambda x:-(x[0]+x[1]+x[2])/C_ref+SOC-SOC_low},
            {'type':'ineq','fun':lambda x:-(x[0]+x[1])/C_ref+SOC-SOC_low},
            {'type':'ineq','fun':lambda x:-(x[0])/C_ref+SOC-SOC_low},
            
            
            {'type':'ineq','fun':lambda x:ch_rate-x[0]},
            {'type':'ineq','fun':lambda x:ch_rate-x[1]},
            {'type':'ineq','fun':lambda x:ch_rate-x[2]},
            {'type':'ineq','fun':lambda x:ch_rate-x[3]},
            {'type':'ineq','fun':lambda x:ch_rate-x[4]},
            
            {'type':'ineq','fun':lambda x:ch_rate+x[0]},
            {'type':'ineq','fun':lambda x:ch_rate+x[1]},
            {'type':'ineq','fun':lambda x:ch_rate+x[2]},
            {'type':'ineq','fun':lambda x:ch_rate+x[3]},
            {'type':'ineq','fun':lambda x:ch_rate+x[4]},
            
            )
    
    x0 = np.array((50,50,50,50,50)) #设置xyz的初始值
    res = minimize(fun,x0,method='COBYLA',constraints=cons)
    return res.x






