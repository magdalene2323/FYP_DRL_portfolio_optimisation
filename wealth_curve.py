#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 11:06:57 2024

@author: magdalenelim
"""
import pandas as pd
import numpy as np
import quantstats as qs
#import riskfolio as rp
from meanvar import compute_meanvar

import matplotlib
import matplotlib.pyplot as plt 
matplotlib.use('Agg')
#%matplotlib inline

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # silence error lbmp5md.dll



dir_ = '00 MAIN DATA/'
def get_daily_return(df):
    df['daily_return']=df.account_value.pct_change(1)    #shift=1
    #Compute daily return: exactly same as SP500['daily_return'] = (SP500['sp500']/ SP500['sp500'].shift(1)) -1
    #df=df.dropna()
    print('Sharpe: ',(252**0.5)*df['daily_return'].mean()/ df['daily_return'].std())
    return df

def get_account_value(model_name, rebalance_window, validation_window, 
                      unique_trade_date, df_trade_date, dir_ = dir_):
    df_account_value = pd.read_csv(f'results/account_value_trade_{model_name}') 
    df_account_value=df_account_value.join(df_trade_date[63:].reset_index(drop=True))
    
    print(" df account value: ", df_account_value )
    return df_account_value

def mean_var_return(daily_return_df):
    plt.style.use('ggplot') #Change/Remove This If you Want

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(daily_return_df.mean(axis=1), alpha=0.5, color='red', label='mean cumulative return', linewidth = 1.0)
    print(daily_return_df.index.values)
    ax.fill_between(daily_return_df.index.values, daily_return_df.mean(axis=1) - daily_return_df.std(axis=1), daily_return_df.mean(axis=1) + daily_return_df.std(axis=1), color='#888888', alpha=0.4)
    ax.fill_between(daily_return_df.index.values, daily_return_df.mean(axis=1) - 2*daily_return_df.std(axis=1), daily_return_df.mean(axis=1) + 2*daily_return_df.std(axis=1), color='#888888', alpha=0.2)
    ax.legend(loc='best')
    #ax.set_ylim([-0.04,0.04])
    ax.set_ylabel("Cumulative Returns")
    ax.set_xlabel("Time")

def annualized_sharpe(returns_df: pd.DataFrame, risk_free_rate: float) -> pd.Series:
    daily_returns = returns_df.mean()
    daily_volatility = returns_df.std()
    sharpe_ratio = (daily_returns - risk_free_rate) / daily_volatility
    annualized_sharpe = np.sqrt(252) * sharpe_ratio
    return annualized_sharpe

######################################## START ################################

df=pd.read_csv('data/dow_30_2009_2020.csv')

rebalance_window = 63
validation_window = 63
unique_trade_date = df[(df.datadate > 20151001)&(df.datadate <= 20200707)].datadate.unique()
df_trade_date = pd.DataFrame({'datadate':unique_trade_date})

returns_df = None

'''
algo = 'cpt' #'ensemble' # 'cpt'
actor_mode, cpt_type = 'cpt', 88 #None, None #'cpt', 88 | 'cpt', 95
ent_coef = 1 #"auto"
k = 0 #0 #1
'''

timesteps = 30000
timesteps_auto = 6000 

dji_acct_val = None
meanvar_acct_val = None
dji_rets = None
meanvar_rets = None

color = {'a2c_auto_f': 'red', 'a2c_auto_r': 'orange', 'a2c': 'green', 
         'ppo': 'blue', #'ppo_Intrinsic': 'magenta', 
         'mean-var': 'tab:olive', 'dji': 'black'}
linestyle = ['-', '--', '-.', ':']

minSharpe = .45
maxSharpe = 1.65

dir_ = '/Users/magdalenelim/Documents/GitHub/FYP_DRL_portfolio_optimisation/'

for seed in [6280]:

    plt.figure(figsize=(25, 7))
    
    for algo in ['a2c_auto_f', 'a2c_auto_r', 'a2c', 'ppo']: 
        for model_type in ['Intrinsic', 'Original']: 
    
            if algo == 'a2c_auto_f': # autoregressive with FNN 
                styleID = 0
                name= f'results/auto_A2C_F/seed{seed}/account_value_trade_{model_type}T{timesteps_auto//1000}S{seed}.csv'
                        
                df = get_account_value(dir_+name, rebalance_window, validation_window, 
                                                          unique_trade_date, df_trade_date )
                df.columns = ['account_value','Date']
                df['Date'] = pd.to_datetime(df['Date'].astype(str), format='%Y%m%d')
                df.set_index('Date', inplace=True)
                
                
                algoname = f'{algo}_{model_type}'
               
                algo_rets = df['account_value']
                algo_rets = df.ffill()
                algo_rets = df.pct_change(1)
                sharpe = annualized_sharpe(algo_rets, 0)
                
                sharpe_norm = (sharpe - minSharpe) / (maxSharpe - minSharpe)
                
                print(sharpe, sharpe_norm)
                
                plt.plot(df, color = color[algo], alpha = sharpe_norm, linestyle = linestyle[styleID],
                         label = algoname)
                
                styleID += 1
                
            elif algo == 'a2c_auto_r': # autoregressive with RNN 
                styleID = 0
                name= f'results/auto_A2C_R/seed{seed}/account_value_trade_{model_type}T{timesteps_auto//1000}S{seed}.csv'
                        
                df = get_account_value(dir_+name, rebalance_window, validation_window, 
                                                          unique_trade_date, df_trade_date )
                df.columns = ['account_value','Date']
                df['Date'] = pd.to_datetime(df['Date'].astype(str), format='%Y%m%d')
                df.set_index('Date', inplace=True)
                
                algoname = f'{algo}_{model_type}'
                algo_rets = df['account_value']
                algo_rets = df.ffill()
                algo_rets = df.pct_change(1)
                
                sharpe = annualized_sharpe(algo_rets, 0)   
                sharpe_norm = (sharpe - minSharpe) / (maxSharpe - minSharpe)
                
                print(sharpe, sharpe_norm)
                
                plt.plot(df, color = color[algo], alpha = sharpe_norm, linestyle = linestyle[styleID],
                         label = algoname)
                
                styleID += 1
                
            
            
            elif algo == 'a2c': # non-autoregressive 
                styleID = 0
                name= f'results/A2C/seed{seed}/account_value_trade_{model_type}T{timesteps_auto//1000}S{seed}.csv'
                df = get_account_value(dir_+name, rebalance_window, validation_window, 
                                                          unique_trade_date, df_trade_date )
                df.columns = ['account_value','Date']
                df['Date'] = pd.to_datetime(df['Date'].astype(str), format='%Y%m%d')
                df.set_index('Date', inplace=True)
            
                algoname = f'{algo}_{model_type}' 
                    
                    
                algo_rets = df['account_value']
                algo_rets = algo_rets.ffill()
                algo_rets = algo_rets.pct_change(1)
                
                sharpe = annualized_sharpe(algo_rets, 0)
                sharpe_norm = (sharpe - minSharpe) / (maxSharpe - minSharpe)
    
                print(sharpe, sharpe_norm)
                
                plt.plot(df, color = color[algoname], alpha = sharpe_norm, linestyle = linestyle[styleID],
                         label = algoname + ', Sharpe={}'.format(np.round(sharpe, 2)))
                
                styleID += 1

        
            elif algo == 'mean-var':
                if meanvar_acct_val is None:
                    meanvar_rets, meanvar_acct_val = compute_meanvar()
                
                algo_rets = meanvar_rets
                sharpe = annualized_sharpe(algo_rets, 0)
                sharpe_norm = (sharpe - minSharpe) / (maxSharpe - minSharpe)
    
                print(sharpe, sharpe_norm)
                plt.plot(meanvar_acct_val, color = color[algo], alpha = sharpe_norm,
                         label = 'Mean-Var, Sharpe={}'.format(np.round(sharpe, 2)))
                    
            
            elif algo == 'dji':
                
                if dji_acct_val is None:
    
                    dji_df = pd.read_csv("data/^DJI.csv")
                    dji_df['Date'] = pd.to_datetime(dji_df['Date'], format='%d/%m/%Y')
                    dji_df.set_index('Date', inplace=True)
                    
                    dji_rets = dji_df['Adj Close'].pct_change(1)
                    dji_rets = dji_rets.ffill()
                    #dji_rets = dji_rets.pct_change(1)
                    #dji_rets[0] = 0
                    #dji_rets += 1
                    #dji_rets.index # from 2009
                    #tqc_cpt_actor.index # from 2016
    
                    dji_rets = dji_rets[df.index] # check 
                    dji_rets[0] = 0
                    print(annualized_sharpe(dji_rets, risk_free_rate=0))
    
                    dji_acct_val = dji_rets + 1
                    dji_acct_val = dji_acct_val.cumprod()
                    dji_acct_val *= 1e6
                                        
                algo_rets = dji_rets
                sharpe = annualized_sharpe(algo_rets, 0)
                sharpe_norm = (sharpe - minSharpe) / (maxSharpe - minSharpe)
    
                print(sharpe, sharpe_norm)
                
                plt.plot(dji_acct_val, color = color[algo], alpha = sharpe_norm,
                         label = 'DJI, Sharpe={}'.format(np.round(sharpe, 2)))
            
        
            else:
                raise NotImplementedError()
            
# =============================================================================
#             if returns_df is None:
#                 returns_df = pd.DataFrame(index=tqc_cpt_actor.index)
# =============================================================================
            
 
        
    plt.legend()
    plt.ylim([.9*1e6, 2.*1e6]) 
    plt.xlim([df.index[0], df.index[-1]])
    plt.savefig('./output/wealth_curve_S{}.png'.format(seed), dpi=200)
    plt.close()
    


'''
#########

dji_rets = dji_df['Adj Close'].pct_change(1)
dji_rets = dji_rets.ffill()
#dji_rets = dji_rets.pct_change(1)
#dji_rets[0] = 0
#dji_rets += 1
#dji_rets.index # from 2009
#tqc_cpt_actor.index # from 2016

dji_rets = dji_rets[tqc_cpt_actor.index]
dji_rets[0] = 0
print(annualized_sharpe(dji_rets, risk_free_rate=0))

dji_acct_val = dji_rets + 1
dji_acct_val = dji_acct_val.cumprod()
dji_acct_val *= 1e6

# plt.plot(dji_acct_val)

######

dji_df['dji_returns'] = dji_df['Adj Close'].pct_change(1) #Compute daily returns of DJIA
dji_df = dji_df.dropna()
dji_df.tail()
#returns_df['dji'] = dji_df['Adj Close']
'''

returns_df = returns_df.ffill()
returns_df = returns_df.pct_change(1)

#annualized_sharpe(returns_df, risk_free_rate=0)

'''
##################

alpha_ = 0.95
lambda_ = 1.5
rho1, rho2 = 0.5, 0.5
b_ = 0

def compute_CPT(tensor, sort = True, B=b_, alpha_=alpha_, lambda_ = lambda_,
                rho1 = rho1, rho2 = rho2):
   
    #print('inside compute_CPT, params:', alpha_, rho1, lambda_, B)
   
    if sort:
        tensor, _ = th.sort(tensor)

    quantiles = tensor
    
    utilities = th.where(quantiles >= B, ((quantiles-B).abs())**alpha_, -lambda_ * ((B-quantiles).abs())**alpha_)
    
    batchSize = tensor.shape[0] #1 # default, by implementation
    supportSize = quantiles.shape[-1] # len(quantiles)
    supportTorch = th.linspace(0.0, 1.0, 1 + supportSize)
    tausPos1 = (1 - supportTorch[:-1]).repeat(batchSize, 1).view(batchSize, 1, -1) # dim = ???
    tausPos2 = (1 - supportTorch[1:]).repeat(batchSize, 1).view(batchSize, 1, -1)
    tausNeg1 = supportTorch[1:].repeat(batchSize, 1).view(batchSize, 1, -1)
    tausNeg2 = supportTorch[:-1].repeat(batchSize, 1).view(batchSize, 1, -1)
    
    weightedProbs = th.where(quantiles >= B, 
                             tausPos1**rho1 / ((tausPos1**rho1 + (1-tausPos1)**rho1)**(1/rho1)) - tausPos2**rho1 / ((tausPos2**rho1 + (1-tausPos2)**rho1)**(1/rho1)), 
                             tausNeg1**rho2 / ((tausNeg1**rho2 + (1-tausNeg1)**rho2)**(1/rho2))  - tausNeg2**rho2 / ((tausNeg2**rho2 + (1-tausNeg2)**rho2)**(1/rho2)))

    CPT_val = (utilities * weightedProbs).sum(-1) #.sum(2)

    return CPT_val # dim: (batchSize, 1)

##################
'''

def assess_pf(df):
    idx = list(df.columns)
    res_df = pd.DataFrame(index=idx, columns=['Sharpe', 'Cum Rets', 'CAGR', 'Ann Vol', 'Max DD'])
    #, 'Alpha', 'Beta'])
    
    #res_df = pd.DataFrame(index=idx, columns=['CPT95', 'CPT88', 'Sharpe', 'Cum Rets', 'CAGR', 'Ann Vol', 'Max DD'])
    
    for col in df.columns:
        row = []
        
        #print(df)
        #print(th.tensor(df[col]*252*100)) # (1, 25)
        
        '''
        cpt88= compute_CPT(th.tensor(df[col][1:]*252*100).view(1, -1), alpha_=.88, lambda_=2.25, 
                         rho1=.65, rho2=.65)[0][0].item()
        cpt95= compute_CPT(th.tensor(df[col][1:]*252*100).view(1, -1), alpha_=.95, lambda_=1.5, 
                         rho1=.5, rho2=.5)[0][0].item()
        
        print(cpt88)
        '''
        sharpe = qs.stats.sharpe(df[col])
        cum_rets = qs.stats.comp(df[col]) 
        cagr = qs.stats.cagr(df[col])   
        vol = qs.stats.volatility(df[col])   
        max_drawdown = qs.stats.max_drawdown(df[col])
        #greeks = qs.stats.greeks(df[col], df['dji']) 
        #alpha, beta = greeks['alpha'], greeks['beta']
        
        row = sharpe, cum_rets, cagr, vol, max_drawdown#, alpha, beta
        #row = cpt88, cpt95, sharpe, cum_rets, cagr, vol, max_drawdown#, alpha, beta
        res_df.loc[col, :] = row
        
    return res_df.astype(float)

res_df = assess_pf(returns_df)

res_df[["Cum Rets","CAGR","Ann Vol","Max DD"]] *=100
#res_df = res_df.round(2)

print(res_df.round(2))
print('\n')
print(res_df.describe().round(2))
'''
print('MEAN:', res_df.iloc[:-1, :].mean())
print('MIN:', res_df.iloc[:-1, :].std())
print('MAX:', res_df.iloc[:-1, :].std())
print('STDEV:', res_df.iloc[:-1, :].std())
'''