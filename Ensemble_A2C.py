# common library
import pandas as pd
import numpy as np
import time
import gym
from gym import spaces
from gym.utils import seeding

import time
import csv    
import sys 
sys.path.append("/Users/magdalenelim/Desktop/FYP")




from stable_baselines import A2C

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy, CnnPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.common.vec_env import DummyVecEnv
from preprocessing.preprocessors import *
from config import config

# customized env
from env.EnvMultipleStock_train import StockEnvTrain
from env.EnvMultipleStock_validation import StockEnvValidation
from env.EnvMultipleStock_trade import StockEnvTrade
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # silence error lbmp5md.dll

from preprocessing.preprocessors import data_split
import quantstats as qs




def metrics(value): 
    rets = value.pct_change(1) 
    sharpe = (252 ** 0.5) * rets.mean() / rets.std()
    vol = (rets.std()*np.sqrt(252))*100
    cagr = ((value.iloc[-1]/value.iloc[0])**(1/(len(value.index)/252))-1)*100
    max_dd = qs.stats.max_drawdown(value )
    return sharpe, vol, cagr, max_dd 



def train_A2C_intrinsic(env_train, model_name, timesteps, v_mix_coef, r_in_coef, v_ex_coef, ent_coef, lr_alpha, lr_beta, rms_prop_eps, verbose, seed,preproc, n_steps,start_intrinsic_update):
    start = time.time()
    
    if preproc:
        
            preproc_ = A2C('MlpPolicyIntrinsicInnovationReward', env_train, v_mix_coef=v_mix_coef, v_ex_coef=v_ex_coef, r_in_coef=r_in_coef, ent_coef=ent_coef,
                        lr_alpha=lr_alpha, lr_beta=lr_beta, epsilon = rms_prop_eps, verbose=verbose, seed=seed, n_steps=1600, 
                        preproc=False, model_type='intrinsic')
            preproc_.learn(total_timesteps=1600)
            
            avg_feature = np.array([np.mean(preproc_.mb_obs[:,d]) for d in range(181)], dtype=np.float64)
            std_feature_ = np.array([np.std(preproc_.mb_obs[:, d]) for d in range(181)])
            std_feature = np.array(1/(1 + std_feature_), dtype=np.float64) 
            
    model = A2C('MlpPolicyIntrinsicInnovationReward', env_train, v_mix_coef=v_mix_coef, v_ex_coef=v_ex_coef, r_in_coef=r_in_coef, ent_coef=ent_coef,
                lr_alpha=lr_alpha, lr_beta=lr_beta, epsilon = rms_prop_eps, verbose=verbose, seed=seed, n_steps=n_steps, 
                preproc=preproc, avg_feat=avg_feature,std_feat=std_feature, model_type='intrinsic')
    
    print("TRAINING....")
    
    model.learn(total_timesteps=timesteps, start_intrinsic_update =start_intrinsic_update)
    
   
    
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (A2C_rnn): ', (end - start) / 60, ' minutes')
    return model





def train_A2C_no_intrinsic(env_train, model_name, timesteps, v_mix_coef, v_ex_coef, ent_coef, lr_alpha, lr_beta, rms_prop_eps, verbose, seed,preproc, n_steps):
    start = time.time()

    if preproc:
            preproc_ = A2C('MlpPolicyIntrinsicInnovationReward', env_train, v_mix_coef=v_mix_coef, v_ex_coef=v_ex_coef, r_in_coef=0, ent_coef=ent_coef,
                        lr_alpha=lr_alpha, lr_beta=lr_beta, epsilon = rms_prop_eps, verbose=verbose, seed=seed, n_steps=1600, 
                        preproc=False, model_type='original')
            preproc_.learn(total_timesteps=1600)
            
            avg_feature = np.array([np.mean(preproc_.mb_obs[:,d]) for d in range(181)], dtype=np.float64)
            std_feature_ = np.array([np.std(preproc_.mb_obs[:, d]) for d in range(181)])
            std_feature = np.array(1/(1 + std_feature_), dtype=np.float64) 
            
    
    model = A2C('MlpPolicyIntrinsicInnovationReward', env_train, v_mix_coef=v_mix_coef, v_ex_coef=v_ex_coef,r_in_coef=0, ent_coef=ent_coef,
                lr_alpha=lr_alpha, lr_beta=lr_beta, epsilon = rms_prop_eps, verbose=verbose, seed=seed, n_steps=n_steps, 
                preproc=preproc,  avg_feat=avg_feature,std_feat=std_feature,model_type='original')
    
    
    print("TRAINING....")
    
    model.learn(total_timesteps=timesteps,start_intrinsic_update=100000)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (A2C): ', (end - start) / 60, ' minutes')
    return model








def DRL_intrinsic_prediction(df,
                   model,
                   name,
                   last_state,
                   iter_num,
                   unique_trade_date,
                   rebalance_window,
                   turbulence_threshold,
                   initial ):
    
    trade_data = data_split(df, start=unique_trade_date[iter_num - rebalance_window ], end=unique_trade_date[iter_num])
    env_trade = DummyVecEnv([lambda: StockEnvTrade(trade_data,
                                                   turbulence_threshold=turbulence_threshold,
                                                   initial=initial,
                                                   previous_state=last_state, 
                                                   model_name=name,
                                                   iteration=iter_num)])
    

    obs_trade = env_trade.reset() 
    dones =[False]
    
    for i in range(len(trade_data.index.unique())):
        action, _ = model.predict_intrinsic(obs_trade)
        obs_trade, rewards, dones, info = env_trade.step(action)
    
    
        if i == (len(trade_data.index.unique()) - 2):
            last_state = env_trade.env_method("render")[0] 
            
            
    
    
            
    
    
    df_last_state = pd.DataFrame({'last_state': last_state})
    df_last_state.to_csv('results/last_state_{}_{}.csv'.format(name, i), index=False)
    return last_state 

def DRL_intrinsic_validation(model, test_data, test_env, test_obs) -> None:
    for i in range(len(test_data.index.unique())):
        action, _states = model.predict_intrinsic(test_obs)
        test_obs, rewards, dones, info = test_env.step(action) 
        


def get_validation_sharpe(iteration):
    df_total_value = pd.read_csv(f'results/account_value_validation_{iteration}.csv', index_col=0)
    df_total_value.columns = ['account_value_train']
    df_total_value['daily_return'] = df_total_value.pct_change(1)
    sharpe = (252 ** 0.5) * df_total_value['daily_return'].mean() / \
             df_total_value['daily_return'].std()
             
    return sharpe



def run_a2c_ensemble_strategy(df, unique_trade_date, rebalance_window, validation_window, timesteps,v_mix_coef, v_ex_coef, r_in_coef, ent_coef, lr_alpha, lr_beta, 
                              rms_prop_eps, verbose, seed, turb_var, preproc, n_steps, start_intrinsic_update):
    print("============Start A2C Strategy============")
    unique_train_date = df[(df.datadate > 20081231)&(df.datadate <= 20200707)].datadate.unique()

    last_state_ensemble, last_state_a2c, last_state_org, model_use =[], [] , [] , []



    # based on the analysis of the in-sample data
    #turbulence_threshold = 140
    insample_turbulence = df[(df.datadate<20151000) & (df.datadate>=20090000)]
    insample_turbulence = insample_turbulence.drop_duplicates(subset=['datadate'])
    insample_turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, turb_var)

    start = time.time()
    for i in range(rebalance_window + validation_window, len(unique_trade_date), rebalance_window):
        
        print("ITER NUMBER: ", i )
        print("============================================")
        ## initial state is empty
        if i - rebalance_window - validation_window == 0:
            # inital state
            initial = True
        else:
            # previous state
            initial = False
            
          

        # Tuning trubulence index based on historical data
        # Turbulence lookback window is one quarter
        end_date_index = df.index[df["datadate"] == unique_trade_date[i - rebalance_window - validation_window]].to_list()[-1]
        
        
        start_date_index = end_date_index - validation_window*30 + 1

        historical_turbulence = df.iloc[start_date_index:(end_date_index + 1), :]
        #historical_turbulence = df[(df.datadate<unique_trade_date[i - rebalance_window - validation_window]) & (df.datadate>=(unique_trade_date[i - rebalance_window - validation_window - 63]))]


        historical_turbulence = historical_turbulence.drop_duplicates(subset=['datadate'])

        historical_turbulence_mean = np.mean(historical_turbulence.turbulence.values)
        

        if historical_turbulence_mean > insample_turbulence_threshold:
            # if the mean of the historical data is greater than the 90% quantile of insample turbulence data
            # then we assume that the current market is volatile,
            # therefore we set the 90% quantile of insample turbulence data as the turbulence threshold
            # meaning the current turbulence can't exceed the 90% quantile of insample turbulence data
            turbulence_threshold = insample_turbulence_threshold
        else:
            # if the mean of the historical data is less than the 90% quantile of insample turbulence data
            # then we tune up the turbulence_threshold, meaning we lower the risk
            turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, 1)
        print("turbulence_threshold: ", turbulence_threshold)

        ############## Environment Setup starts ##############
        ## training env
        # Note: since StockEnvTrain() does NOT specify seed
        # seed to make random number sequence in the algorithm reproducible
        # By default seed is None which means seed from system noise generator (not
        if initial: 
            start_date = 20090101
            start_id=0 
      
        else: 
            start_id += 63 
            start_date = unique_train_date[start_id]
        
        train = data_split(df, start=start_date, end=unique_trade_date[i - rebalance_window - validation_window])
        env_train = DummyVecEnv([lambda: StockEnvTrain(train,preproc=True )])
        
        
        print("train", start_date, unique_trade_date[i - rebalance_window - validation_window])

        
        validation = data_split(df, start=unique_trade_date[i - rebalance_window - validation_window],
                                end=unique_trade_date[i - rebalance_window])
        
        print("val",unique_trade_date[i - rebalance_window - validation_window], unique_trade_date[i - rebalance_window] )
        
        
        
        env_val = DummyVecEnv([lambda: StockEnvValidation(validation,  
                                                          turbulence_threshold=turbulence_threshold,
                                                          iteration=i)])
        
        obs_val = env_val.reset()

        ############## Environment Setup ends ##############
  

        ############## Training and Validation starts ##############
        print("======Model training from: ", start_date, "to ",
              unique_trade_date[i - rebalance_window - validation_window])
        
        print("======A2C Intrinsic Training========")     
        
        
        model_a2c = train_A2C_intrinsic(env_train, model_name="A2CIntrinsic_{}".format(i) , timesteps=timesteps, v_mix_coef=v_mix_coef, r_in_coef=r_in_coef, v_ex_coef=v_ex_coef,
                                                ent_coef=ent_coef, lr_alpha = lr_alpha, lr_beta = lr_beta, rms_prop_eps=rms_prop_eps, verbose=verbose, seed=seed, preproc=preproc, n_steps=n_steps, start_intrinsic_update=start_intrinsic_update) 
        
        DRL_intrinsic_validation(model=model_a2c, test_data=validation, test_env=env_val, test_obs=obs_val)
        sharpe_a2c = get_validation_sharpe(i)

                                    
        print("======A2C Original Training========")
        
        env_train = DummyVecEnv([lambda: StockEnvTrain(train,preproc=True)])
        model_a2c_org = train_A2C_no_intrinsic(env_train, model_name="A2COriginal_{}".format(i), timesteps=timesteps, v_mix_coef=v_mix_coef, v_ex_coef=v_ex_coef, 
                                                ent_coef=ent_coef, lr_alpha = lr_alpha,lr_beta=lr_beta, rms_prop_eps=rms_prop_eps, verbose=verbose, seed=seed, preproc=preproc, n_steps=n_steps)
        
        
        
        env_val = DummyVecEnv([lambda: StockEnvValidation(validation,
                                                          turbulence_threshold=turbulence_threshold,
                                                          iteration=i)])
        obs_val = env_val.reset()
        
        
        _ = DRL_intrinsic_validation(model=model_a2c_org, test_data=validation, test_env=env_val, test_obs=obs_val) # NONE 
        sharpe_a2c_org = get_validation_sharpe(i) 

        
        print("======A2C Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
              unique_trade_date[i - rebalance_window])
       
       
    

        # Model Selection based on sharpe ratio
        
        if sharpe_a2c >= sharpe_a2c_org:
            model_ensemble = model_a2c 
            model_use.append('A2C') 

        else:
            model_ensemble = model_a2c_org
            model_use.append('A2C_original')
         
        
        ############## Training and Validation ends ##############

        ############## Trading starts ##############
        print("======Trading from: ", unique_trade_date[i - rebalance_window - 10], "to ", unique_trade_date[i])
        print("Used Model: ", model_ensemble)
        
        last_state_a2c  = DRL_intrinsic_prediction(df=df, model= model_a2c, name=f'IntrinsicT{timesteps//1000}S{seed}',
                                        last_state=last_state_a2c,  iter_num=i,
                                        unique_trade_date=unique_trade_date,
                                        rebalance_window=rebalance_window,
                                        turbulence_threshold=turbulence_threshold,
                                        initial=initial )
        
        last_state_org  = DRL_intrinsic_prediction(df=df, model= model_a2c_org, name=f'OriginalT{timesteps//1000}S{seed}',
                                        last_state=last_state_org ,  iter_num=i,
                                        unique_trade_date=unique_trade_date,
                                        rebalance_window=rebalance_window,
                                        turbulence_threshold=turbulence_threshold,
                                        initial=initial )
        
        last_state_ensemble  = DRL_intrinsic_prediction(df=df, model=model_ensemble, name=f'EnsembleT{timesteps//1000}S{seed}',
                                        last_state=last_state_ensemble,  iter_num=i,
                                        unique_trade_date=unique_trade_date,
                                        rebalance_window=rebalance_window,
                                        turbulence_threshold=turbulence_threshold,
                                        initial=initial )


        
        print("============Trading Done============")
        ############## Trading ends ##############

    intrinsic_value = pd.read_csv(f'results/account_value_trade_IntrinsicT{timesteps//1000}S{seed}.csv',index_col=0, header=None )
    intrin_sharpe, intrin_vol, intrin_cagr, intrin_maxdd = metrics(intrinsic_value[1])
    org_value = pd.read_csv(f'results/account_value_trade_OriginalT{timesteps//1000}S{seed}.csv',index_col=0, header=None )
    org_sharpe, org_vol, org_cagr, org_maxdd = metrics(org_value[1] )
    ensemble_value = pd.read_csv(f'results/account_value_trade_EnsembleT{timesteps//1000}S{seed}.csv',index_col=0, header=None )
    ens_sharpe, ens_vol, ens_cagr, ens_maxdd = metrics(ensemble_value[1] )
    end = time.time()
    infos = [ens_sharpe, ens_vol, ens_cagr, ens_maxdd,intrin_sharpe, intrin_vol, intrin_cagr, intrin_maxdd, org_sharpe, org_vol, org_cagr, org_maxdd ]
    print("A2C Strategy took: ", (end - start) / 60, " minutes")
    return infos , model_use 
    

    
def run_once(path, preproc=True, timesteps=30000, v_mix_coef=0.1, v_ex_coef=1, r_in_coef=0.001, ent_coef=0.01, lr_alpha=7e-4, lr_beta=7e-4, 
             rms_prop_eps=1e-5, verbose=0, seed=51104, turb_var=0.9 , n_steps = 5, start_intrinsic_update=10000):
    
        preprocessed_path = path 
        data = pd.read_csv(preprocessed_path, index_col=0)
        unique_trade_date = data[(data.datadate > 20151001)&(data.datadate <= 20200707)].datadate.unique()
        rebalance_window = 63
        validation_window = 63
        
        
    
        turbID = int(100*turb_var)

        infos , model_use = run_a2c_ensemble_strategy(data, unique_trade_date, rebalance_window, validation_window,
                              timesteps = timesteps, v_mix_coef=v_mix_coef,v_ex_coef=v_ex_coef, r_in_coef=r_in_coef, ent_coef=ent_coef, lr_alpha=lr_alpha, lr_beta=lr_beta,
                              rms_prop_eps=rms_prop_eps, verbose=verbose, seed=seed, turb_var=turb_var, preproc=preproc, n_steps= n_steps ,start_intrinsic_update=start_intrinsic_update)
        
        f=  open(f"/Users/magdalenelim/Desktop/FYP/results/SUMMARYRESULTS.csv", 'a', newline='')
        to_append = [['SEED', seed], infos, model_use ]                 
        csvwriter = csv.writer(f)
        csvwriter.writerows(to_append)
        f.close()




def run(): 
    path='/Users/magdalenelim/Desktop/FYP/done_data.csv'
    for seed in [6280, 43136, 85721,17913,51104,35269,8182,40124,5921,3402,9391,6574,43523,10672,75927]:
        run_once(path,seed=seed) 
    


run() 

    

