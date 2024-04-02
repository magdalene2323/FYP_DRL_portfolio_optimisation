
import time
import sys
import csv

import gym 
from gym import spaces

import tensorflow as tf
import numpy as np 
from itertools import chain

from stable_baselines import logger
from stable_baselines.common import explained_variance, tf_util, ActorCriticRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.policies import ActorCriticPolicy, RecurrentActorCriticPolicy
from stable_baselines.common.runners import AbstractEnvRunner
from stable_baselines.common.schedules import Scheduler, PiecewiseSchedule 
from stable_baselines.common.tf_util import mse, total_episode_reward_logger, calc_entropy
from stable_baselines.common.math_util import safe_mean

from collections import deque

OB_SPACE_SHAPE=181 
STOCK_DIM = 30 

def discount_with_dones(rewards, dones, gamma):    ## Same function with LIRPG/baselines/a2c/utils.py
    """
    Apply the discount value to the reward, where the environment is not done

    :param rewards: ([float]) The rewards
    :param dones: ([bool]) Whether an environment is done or not
    :param gamma: (float) The discount value
    :return: ([float]) The discounted rewards
    """
    discounted = []
    ret = 0  # Return: discounted reward
    for reward, done in zip(rewards[::-1], dones[::-1]):
        ret = reward + gamma * ret * (1. - done)  # fixed off by one bug
        discounted.append(ret)
    return discounted[::-1]


class A2C(ActorCriticRLModel):
    """
    The A2C (Advantage Actor Critic) model class, https://arxiv.org/abs/1602.01783

    :param policy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) Discount factor
    :param n_steps: (int) The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param vf_coef: (float) Value function coefficient for the loss calculation
    :param ent_coef: (float) Entropy coefficient for the loss calculation
    :param max_grad_norm: (float) The maximum value for the gradient clipping
    :param learning_rate: (float) The learning rate
   
    :param alpha: (float)  RMSProp decay parameter (default: 0.99)
    :param momentum: (float) RMSProp momentum parameter (default: 0.0)
    :param epsilon: (float) RMSProp epsilon (stabilizes square root computation in denominator of RMSProp update)
        (default: 1e-5)

    :param lr_schedule: (str) The type of scheduler for the learning rate update ('linear', 'constant',
                              'double_linear_con', 'middle_drop' or 'double_middle_drop')
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
                              (used only for loading)
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly
    :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed. Note that if you want completely deterministic
        results, you must set `n_cpu_tf_sess` to 1.
    :param n_cpu_tf_sess: (int) The number of threads for TensorFlow operations
        If None, the number of cpu of the current machine will be used.
    """
    def __init__(self, policy, env, gamma=0.99, n_steps=5, #vf_coef=0.25,
                 v_mix_coef=0.5, v_ex_coef=1.0, r_in_coef=0.001,
                 ent_coef=0.01, max_grad_norm=0.5,
                 #learning_rate=7e-4,
                 lr_alpha = 7e-4, lr_beta = 7e-4,
                 alpha=0.99, momentum=0.0, epsilon=1e-5, lr_schedule='constant',preproc=False, avg_feat = None, std_feat=None ,
                 verbose=0, tensorboard_log=None, 
                 _init_setup_model=True, policy_kwargs=None,
                 full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None, model_type=None ):
        
        if r_in_coef > 0: 
            print("MODEL: Non-Autoregressive A2C + Intrinsic") 
        else: 
            print("MODEL: Non-Autoregressive A2C (No Intrinsic)") 
#use lr_alpha and lr_beta to replace learning_rate
        self.n_steps = n_steps
        self.gamma = gamma
        # avg_feature, std_feature are np.array of shape (181,) 
        
        
        if preproc: #True 
            avg_feature = avg_feat
            std_feature = std_feat 
        else: 
            avg_feature = np.zeros(OB_SPACE_SHAPE)
            std_feature = np.ones(OB_SPACE_SHAPE)
        
        
        
            
        self.avg_feature = np.reshape(avg_feature, (1, OB_SPACE_SHAPE)) 
        self.std_feature = np.reshape(std_feature,(1, OB_SPACE_SHAPE)) 
        
       

        # self.avg_feature, self.std_feature are np.array of shape (1,181)
        self.train_model_avg_feature = np.tile(self.avg_feature, (self.n_steps, 1)) # (nsteps, 181) 
        self.train_model_std_feature = np.tile(self.std_feature, (self.n_steps, 1)) # (nsteps, 181)
        
        
        #self.vf_coef = vf_coef
        self.v_mix_coef = v_mix_coef
        self.v_ex_coef = v_ex_coef
        self.r_in_coef = r_in_coef
        self.model_type = model_type # intrinsic or original 
        
        
        self.r_ex_coef = 1-r_in_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.alpha = alpha
        self.momentum = momentum
        self.epsilon = epsilon
        self.lr_schedule = lr_schedule
        #self.learning_rate = learning_rate
        self.lr_alpha = lr_alpha
        self.lr_beta = lr_beta
        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log

        #self.learning_rate_ph = None
        self.LR_ALPHA = None
        self.LR_BETA = None
        self.n_batch = None
        self.ADV_EX = None
        self.RET_EX = None
        self.R_EX = None
        self.DIS_V_MIX_LAST = None
        self.V_MIX = None
        self.A = None
        print("LOGGER", logger)
        #self.actions_ph = None
        #A = tf.compat.v1.placeholder(tf.int32, [self.nbatch], 'A')
        #self.advs_ph = None
        #ADV_EX = tf.compat.v1.placeholder(tf.float32, [nbatch], 'ADV_EX')
        #self.rewards_ph = None
        #R_EX = tf.compat.v1.placeholder(tf.float32, [nbatch], 'R_EX')
        #RET_EX = tf.compat.v1.placeholder(tf.float32, [nbatch], 'RET_EX')
        #V_MIX = tf.compat.v1.placeholder(tf.float32, [nbatch], 'V_MIX')
        #DIS_V_MIX_LAST = tf.compat.v1.placeholder(tf.float32, [nbatch], 'DIS_V_MIX_LAST')
        #COEF_MAT = tf.compat.v1.placeholder(tf.float32, [nbatch, nbatch], 'COEF_MAT')  WE DON'T NEED COEF_MAT AND MATMUL(COEF_MAT * REWARDS), REWARD WOULD BE ENOUGH
        #LR_ALPHA = tf.compat.v1.placeholder(tf.float32, [], 'LR_ALPHA')
        #LR_BETA = tf.compat.v1.placeholder(tf.float32, [], 'LR_BETA')
        self.pg_mix_loss = None
        self.pg_ex_loss = None
        self.v_mix_loss = None
        self.v_ex_loss = None
        self.entropy = None
        #self.apply_backprop = None
        self.train_model = None
        self.step_model = None
        self.proba_step = None
        self.value = None
        self.learning_rate_schedule = None
        self.summary = None


        super(A2C, self).__init__(policy=policy, env=env, verbose=verbose, requires_vec_env=True,
                                  _init_setup_model=_init_setup_model, policy_kwargs=policy_kwargs,
                                  seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)

        # if we are loading, it is possible the environment is not known, however the obs and action space are known
        if _init_setup_model:
            self.setup_model()

    def _make_runner(self) -> AbstractEnvRunner:
        return A2CRunner(self.env, self, n_steps=self.n_steps,r_ex_coef=self.r_ex_coef, r_in_coef=self.r_in_coef, gamma=self.gamma)
        # calling A2CRunner resets the train environment 

    def _get_pretrain_placeholders(self):  #not used
        policy = self.train_model
        if isinstance(self.action_space, spaces.Discrete):
            return policy.X, self.A, policy.policy
        return policy.X, self.A, policy.deterministic_action

    def setup_model(self):    # Part of the init in LIRPG A2C
        with SetVerbosity(self.verbose):
            # check if the input policy is in the class of A2C policies
            #assert issubclass(self.policy, ActorCriticPolicy), "Error: the input policy for the A2C model must be an " \
            #                                                    "instance of common.policies.ActorCriticPolicy."

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.set_random_seed(self.seed)
                self.sess = tf_util.make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)    # returns a session that will use <num_cpu> CPU's only
                self.n_batch = self.n_envs * self.n_steps

                #line 55-56: Create step and train models
                n_batch_step = None
                n_batch_train = None
                if issubclass(self.policy, RecurrentActorCriticPolicy):
                    n_batch_step = self.n_envs
                    n_batch_train = self.n_envs * self.n_steps
                    
                step_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1, self.avg_feature, self.std_feature, reuse=False) 
                
                                         #n_batch_step, reuse=False, **self.policy_kwargs)
                # A context manager for defining ops that creates variables (layers).

                with tf.compat.v1.variable_scope("train_model", reuse=True,
                                       custom_getter=tf_util.outer_scope_getter("train_model")):
                    train_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs*self.n_steps, self.n_steps,self.train_model_avg_feature, self.train_model_std_feature,reuse=True)
                 
                    

                
                self.R_EX = tf.compat.v1.placeholder(tf.float32, [None], 'R_EX')
                self.DIS_V_MIX_LAST = tf.compat.v1.placeholder(tf.float32, [self.n_batch], 'DIS_V_MIX_LAST')
                self.COEF_MAT = tf.compat.v1.placeholder(tf.float32, [self.n_batch, self.n_batch], 'COEF_MAT')
                self.V_MIX = tf.compat.v1.placeholder(tf.float32, [None], 'V_MIX')
                

                self.A = tf.compat.v1.placeholder(tf.float32, [None, None], 'A') # used 
               
                r_mix = self.r_ex_coef * self.R_EX + self.r_in_coef * train_model.r_in
                ret_mix = tf.squeeze(tf.matmul(self.COEF_MAT, tf.reshape(r_mix, [self.n_batch, 1])), [1]) + self.DIS_V_MIX_LAST
                adv_mix = ret_mix - self.V_MIX
                
                
                
                self.LR_ALPHA = tf.compat.v1.placeholder(tf.float32, [], name="LR_ALPHA")
                self.LR_BETA = tf.compat.v1.placeholder(tf.float32, [], name="LR_BETA")
                
                
                neglogpac = train_model.pd.neglogp(self.A)
                self.entropy = tf.reduce_mean(train_model.pd.entropy()) 
                
                self.pg_mix_loss = tf.reduce_mean(adv_mix * neglogpac)
                self.v_mix_loss = tf.reduce_mean(mse(tf.squeeze(train_model.v_mix), ret_mix))
                
                #rewards_ph is ret in LIRPGa
                # https://arxiv.org/pdf/1708.04782.pdf#page=9, https://arxiv.org/pdf/1602.01783.pdf#page=4
                # and https://github.com/dennybritz/reinforcement-learning/issues/34
                # suggest to add an entropy component in order to improve exploration.
                # Calculate the loss
                # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss
                # Policy loss
                # L = A(s,a) * -logpi(a|s)

                policy_loss = self.pg_mix_loss - self.ent_coef * self.entropy + self.v_mix_coef * self.v_mix_loss
                with tf.compat.v1.variable_scope("policy_info", reuse=False): 
                    tf.compat.v1.summary.scalar('entropy_loss', self.entropy)
                    tf.compat.v1.summary.scalar('pg_mix_loss', self.pg_mix_loss)
                    tf.compat.v1.summary.scalar('v_mix_loss', self.v_mix_loss)
                    tf.compat.v1.summary.scalar('policy_loss', policy_loss)
                    
                self.params = tf_util.get_trainable_vars("policy")  
                
                
                grads = tf.gradients(policy_loss, self.params)   # Using train_model 
                
                if self.max_grad_norm is not None:  # max_grad_norm defines the maximum gradient, needs to be normalized
                    # Clip the gradients (normalize)
                    grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
                grads_and_vars = list(zip(grads, self.params))  # zip pg and policy params correspondingly, policy_grads_and_vars in LIRPG
                
                trainer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=self.LR_ALPHA, decay=self.alpha,
                                                    epsilon=self.epsilon, momentum=self.momentum) #Initialize optimizer 
                
                self.policy_train = trainer.apply_gradients(grads_and_vars)  
                
                
                rmss = [trainer.get_slot(var, 'rms') for var in self.params]  
                
                
                self.params_new = {}
                for grad, rms, var in zip(grads, rmss, self.params):
                    ms = rms + (tf.square(grad) - rms) * (1 - self.alpha) # wrong in this line
                    self.params_new[var.name] = var - self.LR_ALPHA * grad / tf.sqrt(ms + self.epsilon)  
                    

                self.policy_new = None
                self.policy_new = train_model.policy_new_fn(self.params_new, self.observation_space, self.action_space, self.n_envs*self.n_steps, self.n_steps, self.train_model_avg_feature, self.train_model_std_feature)
            

            
                
                
                
                #INTRINSIC UPDATE
                self.ADV_EX = tf.compat.v1.placeholder(tf.float32, [None], 'ADV_EX') #(n_steps,)
                self.RET_EX = tf.compat.v1.placeholder(tf.float32, [None], 'RET_EX')
                        
                
                """with tf.compat.v1.variable_scope("input_info", reuse=False):
                    tf.compat.v1.summary.scalar('ret_mix', tf.reduce_mean(ret_mix))
                    tf.compat.v1.summary.scalar('ret_ex', tf.reduce_mean(self.RET_EX))
                    #tf.summary.histogram('learning_rate', self.learning_rate_ph)
                    tf.compat.v1.summary.scalar('learning_rate_alpha', tf.reduce_mean(self.LR_ALPHA))
                    tf.compat.v1.summary.scalar('learning_rate_beta', tf.reduce_mean(self.LR_BETA))
                    tf.compat.v1.summary.scalar('adv_mix', tf.reduce_mean(adv_mix))
                    tf.compat.v1.summary.scalar('adv_ex', tf.reduce_mean(self.ADV_EX))
                    if self.full_tensorboard_log:
                        tf.compat.v1.summary.histogram('ret_mix_histogram', ret_mix)
                        #tf.summary.histogram('learning_rate', self.learning_rate_ph)
                        tf.compat.v1.summary.histogram('adv_mix_histogram', adv_mix)
                        if tf_util.is_image(self.observation_space):
                            tf.compat.v1.summary.image('observation', train_model.X)
                        else:
                            tf.compat.v1.summary.histogram('observation', train_model.X)"""
                
                
                neglogpac_new = self.policy_new.pd.neglogp(self.A) #shape: (n_steps,) 
                
                ratio_new = tf.exp(tf.stop_gradient(neglogpac) - neglogpac_new)
                self.pg_ex_loss = tf.reduce_mean(-self.ADV_EX * ratio_new)
                self.v_ex_loss = tf.reduce_mean(mse(tf.squeeze(train_model.v_ex), self.RET_EX))
                intrinsic_loss = self.pg_ex_loss + self.v_ex_coef * self.v_ex_loss 
                
                
                
                self.intrinsic_params = tf_util.get_trainable_vars("intrinsic")  #A list of trainable variables [var1, var2, ....]
                
                self.intrinsic_vf_params = tf_util.get_trainable_vars("intrinsic")[0:6]
                

                
                
                
                

                
                
                with tf.compat.v1.variable_scope("intrinsic_info", reuse=False):
                    tf.compat.v1.summary.scalar('pg_ex_loss', self.pg_ex_loss)
                    tf.compat.v1.summary.scalar('v_ex_loss', self.v_ex_loss)
                    tf.compat.v1.summary.scalar('intrinsic_loss', intrinsic_loss)
                    
 
            
                intrinsic_grads = tf.gradients(intrinsic_loss, self.intrinsic_params)
                
                intrinsic_grads1 = tf.gradients(self.v_ex_loss, self.intrinsic_vf_params)
                
        

                if self.max_grad_norm is not None:
                    
                    intrinsic_grads, _ = tf.clip_by_global_norm(intrinsic_grads, self.max_grad_norm)
                    intrinsic_grads1, _ = tf.clip_by_global_norm(intrinsic_grads1, self.max_grad_norm)
                                        
                intrinsic_grads_and_vars = list(zip(intrinsic_grads, self.intrinsic_params))
                intrinsic_grads_and_vars1 = list(zip(intrinsic_grads1, self.intrinsic_vf_params))
                

                intrinsic_trainer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=self.LR_BETA, decay=self.alpha, 
                                                          epsilon=self.epsilon, momentum=self.momentum)
                
                
                intrinsic_trainer1 = tf.compat.v1.train.RMSPropOptimizer(learning_rate=self.LR_BETA, decay=self.alpha, 
                                                          epsilon=self.epsilon, momentum=self.momentum)

                self.intrinsic_train = intrinsic_trainer.apply_gradients(intrinsic_grads_and_vars)
                
                self.intrinsic_train1 = intrinsic_trainer1.apply_gradients(intrinsic_grads_and_vars1)
                
                
            
            
                
                # line 150-159
                self.train_model = train_model
                
                
                
                
                self.step_model = step_model
                self.step = step_model.step 
                self.value = step_model.value
                self.intrinsic_reward =step_model.intrinsic_reward
                self.initial_state = step_model.initial_state
                
                tf.compat.v1.global_variables_initializer().run(session=self.sess)
                
        
                self.summary = tf.compat.v1.summary.merge_all()
            
        
    def _train_step(self,obs, obs_nx, states, actions, r_ex, ret_ex, v_ex, v_mix,
                                                         dis_v_mix_last, coef_mat, update,writer=None,update_intrinsic=True):

        """
        applies a training step to the model

        :param obs: ([float]) The input observations
        :param states: ([float]) The states (used for recurrent policies)
        :param rewards: ([float]) The rewards from the environment
        :param masks: ([bool]) Whether or not the episode is over (used for recurrent policies)
        :param actions: ([float]) The actions taken
        :param values: ([float]) The logits values
        :param update: (int) the current step iteration
        :param writer: (TensorFlow Summary.writer) the writer for tensorboard
        :return: (float, float, float) policy loss, value loss, policy entropy
        """
        #line 120-136: train() in LIRPG, make the training part (feedforward and retropropagation of gradients)
        #advs = rewards - values
        advs_ex = ret_ex - v_ex
        #cur_lr = None
        for _ in range(len(obs)):
            #cur_lr = self.learning_rate_schedule.value()
            cur_lr_alpha = self.lr_alpha.value()            
            cur_lr_beta = self.lr_beta.value()
        assert cur_lr_alpha is not None, "Error: the observation input array cannon be empty"
        assert cur_lr_beta is not None, "Error: the observation input array cannon be empty"
        
        
        
        td_map = {self.train_model.X: obs, self.A: actions, self.train_model.A_ALL: actions, self.train_model.X_NX: obs_nx,self.policy_new.X:obs,
                  self.ADV_EX: advs_ex, self.RET_EX:ret_ex, self.R_EX: r_ex, self.V_MIX:v_mix, self.DIS_V_MIX_LAST:dis_v_mix_last, self.COEF_MAT:coef_mat, 
                  self.LR_ALPHA:cur_lr_alpha, self.LR_BETA:cur_lr_beta} 
        
        
       
        if states is not None:
            td_map[self.train_model.S] = states

    
        
        if update_intrinsic:
            pg_ex_loss, pg_mix_loss, value_mix_loss, value_ex_loss, policy_entropy, _, _  = self.sess.run(
                    [self.pg_ex_loss, self.pg_mix_loss, self.v_mix_loss, self.v_ex_loss, self.entropy, self.policy_train, self.intrinsic_train ], td_map) 
        
        else: 
            pg_mix_loss, value_mix_loss,value_ex_loss, policy_entropy, _ , _ = self.sess.run(
                    [self.pg_mix_loss, self.v_mix_loss, self.v_ex_loss, self.entropy, self.policy_train, self.intrinsic_train1 ], td_map) 
            
            pg_ex_loss =0
                
        
        return pg_ex_loss, pg_mix_loss, value_mix_loss, value_ex_loss, policy_entropy 
    
            
            


    def learn(self, total_timesteps, start_intrinsic_update=50000, callback=None, log_interval=100, tb_log_name="A2C",
              reset_num_timesteps=True):
        ## NO NEED TO HAVE SO MANY INPUT VALUES BECAUSE MODEL IS NOT INITIATED HERE??
        new_tb_log = self._init_num_timesteps(reset_num_timesteps) # self.num_timesteps = 0 
        callback = self._init_callback(callback)
        

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
                    
            self._setup_learn()
            self.lr_alpha = Scheduler(initial_value=self.lr_alpha, n_values=total_timesteps, schedule=self.lr_schedule)
            self.lr_beta = Scheduler(initial_value=self.lr_beta, n_values=total_timesteps, schedule=self.lr_schedule)
            

            t_start = time.time()
            self.ep_info_buf = deque(maxlen=100)
            self.eprexbuf = deque(maxlen=100)
            self.eprinbuf = deque(maxlen=100)
            self.eplenbuf = deque(maxlen=100)
            callback.on_training_start(locals(), globals())
            
            
    
            
            if total_timesteps != 1600: 
                df= self.env.get_attr('df', indices=0)
                df_len = len(df[0].index.unique())
                total_timesteps = total_timesteps + (df_len - total_timesteps % df_len ) 
                print("total timesteps: ", total_timesteps ,'epochs' , total_timesteps/df_len )
            

            
            
            # A2C INTRINSIC TRAINING 

            for update in range(1, total_timesteps // self.n_batch + 1): 
                
                if update % 1000 == 0: 
                    print("No. of days/ updates: ", update, "steps", update*30 )
                
                
                if update < start_intrinsic_update//self.n_batch: 
                    update_intrinsic=False 
                else: 
                    update_intrinsic = True 
                
                
                    
                
                # Get mini batch of experiences
                callback.on_rollout_start()
                rollout = self.runner.run(callback)
               
                       
                obs, actions, obs_nx, states, r_in, r_ex, ret_ex, ret_mix, \
                v_ex, v_mix, last_v_ex, last_v_mix, dones, \
                ep_info, ep_r_ex, ep_r_in, ep_len, v_mix_terminal = rollout 
                
                
                if total_timesteps == 1600: 
                    self.mb_obs = obs 
                    return self 
                
        
                
                if True in dones: 
                    true_index = np.where(dones)[0]
               
                dis_v_mix_last = np.zeros([self.n_batch], np.float32)
                coef_mat = np.zeros([self.n_batch, self.n_batch], np.float32)
                
                
                for i in range(self.n_batch):

                    dis_v_mix_last[i] = self.gamma ** (self.n_steps - i % self.n_steps) * last_v_mix[i // self.n_steps] 
                    coef = 1.0
                    for j in range(i, self.n_batch):  
                        if j > i and j % self.n_steps == 0:
                            break
                        coef_mat[i][j] = coef
                        coef *= self.gamma
                        if dones[j]:
                            dis_v_mix_last[i] = self.gamma ** (true_index - i + 1 ) * v_mix_terminal[0] # Treat terminal obs as end-of-episode obs
                            
                            break
                        

                        
                callback.update_locals(locals())
                callback.on_rollout_end()

                # Early stopping due to the callback
                if not self.runner.continue_training:
                    break

                self.ep_info_buf.extend(ep_info)
                self.eprexbuf.extend(ep_r_ex)
                self.eplenbuf.extend(ep_len)
                
                
                pg_ex_loss, pg_mix_loss, value_mix_loss, value_ex_loss, policy_entropy  = self._train_step(obs, obs_nx, states, actions, r_ex, ret_ex, v_ex, v_mix,
                                                                 dis_v_mix_last, coef_mat, self.num_timesteps // self.n_batch, writer,update_intrinsic)
                
                
                
                
                policy_loss = pg_mix_loss - self.ent_coef * policy_entropy + self.v_mix_coef * value_mix_loss
                
                intrinsic_loss = pg_ex_loss + self.v_ex_coef * value_ex_loss
                
                
                if update % 500 == 0:
                    f = open(f"/Users/magdalenelim/Desktop/FYP/results/{self.model_type}_loss.csv", 'a', newline='')
                    
                    to_append_ = [['update', 'pg_mix_loss','pg_ex_loss', 'v_mix_loss', 'value_ex_loss', 'policy_entropy', 'policy_loss', 'intrinsic_loss','r_in', 'r_ex', 'last_v_mix', 'last_v_ex', 'ret_mix', 'ret_ex', 'actions']]
                    to_append = [[update, pg_mix_loss,pg_ex_loss, value_mix_loss, value_ex_loss, policy_entropy, policy_loss, intrinsic_loss ,r_in, r_ex, last_v_mix, last_v_ex, ret_mix, ret_ex, actions, v_mix_terminal]]                 
                    csvwriter = csv.writer(f)
                    csvwriter.writerows(to_append_)
                    csvwriter.writerows(to_append)
                    f.close()
                

 
                n_seconds = time.time() - t_start
                # Calculate the fps (frame per second)
                fps = int((update * self.n_batch) / n_seconds)


        callback.on_training_end()
        return self

    def save(self, save_path, cloudpickle=False):
        data = {
            "gamma": self.gamma,
            "n_steps": self.n_steps,
            #"vf_coef": self.vf_coef,
            "v_mix_coef": self.v_mix_coef,
            "v_ex_coef": self.v_ex_coef,
            "ent_coef": self.ent_coef,
            "max_grad_norm": self.max_grad_norm,
            #"learning_rate": self.learning_rate,
            "lr_alpha": self.lr_alpha,
            "lr_beta": self.lr_beta,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
            "lr_schedule": self.lr_schedule,
            "verbose": self.verbose,
            "policy": self.policy,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "n_envs": self.n_envs,
            "n_cpu_tf_sess": self.n_cpu_tf_sess,
            "seed": self.seed,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs
        }
        #line 138-141: save() in LIRPG, save the model
        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)





class A2CRunner(AbstractEnvRunner):
    def __init__(self, env, model, n_steps=5,r_ex_coef=1-0.001, r_in_coef=0.001, gamma=0.99):
        """
        A runner to learn the policy of an environment for an a2c model

        :param env: (Gym environment) The environment to learn from
        :param model: (Model) The model to learn
        :param n_steps: (int) The number of steps to run for each environment
        :param gamma: (float) Discount factor
        """
        #line 162-176: needs to modify the parameters in runner; MARKED LINES TO TAKE CARE LATER
        super(A2CRunner, self).__init__(env=env, model=model, n_steps=n_steps)
        self.gamma = gamma

        #self.batch_ob_shape = (self.n_envs*self.n_steps,) + env.observation_space.shape
        #self.obs = env.reset()
        #self.policy_states = model.initial_state
        #self.dones = [False for _ in range(self.n_envs)]
        nenv = env.num_envs
        self.r_ex_coef = r_ex_coef
        self.r_in_coef = r_in_coef
        self.ep_r_in = np.zeros([nenv])
        self.ep_r_ex = np.zeros([nenv])
        self.ep_len = np.zeros([nenv])
        self.states = model.initial_state
        
        
        
        
        

    def _run(self):

        v_mix_terminal= None 
        mb_obs, mb_r_ex, mb_r_in, mb_actions, mb_v_ex, mb_v_mix, mb_dones = [],[],[],[],[],[],[]
        mb_obs_next = []
        mb_states = self.states
        #ep_infos = []
        ep_info, ep_r_ex, ep_r_in, ep_len = [], [], [], []
        i=0 
        while i < self.n_steps : 

            actions,v_ex, v_mix, _, _ = self.model.step(self.obs) 
            mb_obs.append(np.copy(self.obs))
            
            mb_v_mix.append(v_mix)
            mb_dones.append(self.dones)
            clipped_actions = np.clip(actions,  self.env.action_space.low, self.env.action_space.high)
            mb_actions.append(clipped_actions)
            
            
            obs, r_ex, dones, infos = self.env.step(clipped_actions) # dones should be Bool. 
              
            if not dones: 
                i+=1 
                self.model.num_timesteps += self.n_envs 
                
            
            mb_obs_next.append(obs)
            r_in = self.model.intrinsic_reward(ob=self.obs, ac=clipped_actions, ob_nx= obs)    ###EXPAND TO INCULDE OBS_NX, obs = OBS_NX, self.obs = OBS
            
            
            mb_v_ex.append(v_ex)
            mb_r_ex.append(r_ex) 
            mb_r_in.append(r_in)




            if self.callback is not None:
                # Abort training early
                self.callback.update_locals(locals())
                if self.callback.on_step() is False:
                    self.continue_training = False
                    # Return dummy values
                    return [None] * 8

            for info in infos:
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    ep_info.append(maybe_ep_info)

            self.ep_r_ex += r_ex #self.ep_r_ex is a list of 1 value which increases by r_ex every step 
            
            
            self.ep_len += 1

            for n, done in enumerate(dones): # n : which env 
                if done:
                    v_ex_terminal, v_mix_terminal = self.model.value(self.obs)
                    

                    ep_r_ex.append(self.ep_r_ex[n])
                    ep_r_in.append(self.ep_r_in[n])
                    ep_len.append(self.ep_len[n])
                    self.ep_r_ex[n], self.ep_r_in[n], self.ep_len[n] = 0,0,0


            


            self.dones = dones
            self.obs =obs 
            
            
            
             
        mb_dones.append(self.dones)
        mb_dones = mb_dones[1:]
        
        
        if [True] in mb_dones: 
            true_index = mb_dones.index([True])

    
            mb_obs.pop(true_index)
            mb_actions.pop(true_index)
            mb_obs_next.pop(true_index)
            mb_r_ex.pop(true_index)
            mb_r_in.pop(true_index)
            mb_v_ex.pop(true_index)
            mb_v_mix.pop(true_index)
            mb_dones.pop(true_index)
            if true_index>0: 
                mb_dones[true_index-1] = [True]

        
            
        
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype).swapaxes(1, 0).reshape(self.batch_ob_shape) 
        
        mb_obs_nx = np.asarray(mb_obs_next, dtype=self.obs.dtype).swapaxes(1, 0).reshape(self.batch_ob_shape)
        
        

        
        mb_r_ex = np.asarray(mb_r_ex, dtype=np.float32).swapaxes(1, 0) 
        mb_r_in = np.asarray(mb_r_in, dtype=np.float32).swapaxes(1, 0)
        mb_r_mix = self.r_ex_coef * mb_r_ex + self.r_in_coef * mb_r_in
        
    
        
        mb_actions = np.asarray(mb_actions, dtype=self.env.action_space.dtype).swapaxes(0, 1) 
        mb_v_ex = np.asarray(mb_v_ex, dtype=np.float32).swapaxes(1, 0)        
        mb_v_mix = np.asarray(mb_v_mix, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool_).swapaxes(0, 1)
        
        
      
        
        last_v_ex, last_v_mix = self.model.value(self.obs)
        
                        
        mb_ret_ex, mb_ret_mix = np.zeros(mb_r_ex.shape), np.zeros(mb_r_mix.shape)
        
        # discount/bootstrap off value fn
       
        
        
        for n, (r_ex, r_mix, dones, v_ex, v_mix) in enumerate(zip(mb_r_ex, mb_r_mix, mb_dones, last_v_ex, last_v_mix)): 
            r_ex, r_mix = r_ex.tolist(), r_mix.tolist() 
            dones = dones.tolist() 
            if True in dones: 
                r_ex1 = r_ex.copy() 
                r_mix1 = r_mix.copy()
                r_ex1[true_index-1] = r_ex1[true_index-1] + self.gamma*v_ex_terminal[0]  
                r_mix1[true_index-1] = r_mix1[true_index-1] + self.gamma*v_mix_terminal[0]
                ret_ex = discount_with_dones(r_ex1 + [v_ex], dones + [0], self.gamma)[:-1] 
                ret_mix = discount_with_dones(r_mix1 + [v_mix], dones + [0], self.gamma)[:-1]
            else: 
                ret_ex = discount_with_dones(r_ex + [v_ex], dones + [0], self.gamma)[:-1] 
                ret_mix = discount_with_dones(r_mix + [v_mix], dones + [0], self.gamma)[:-1]
         
            
            mb_ret_ex[n], mb_ret_mix[n] = ret_ex, ret_mix
            
            
       

        mb_r_ex = mb_r_ex.flatten()
        mb_r_in = mb_r_in.flatten()
        mb_ret_ex = mb_ret_ex.flatten()
        mb_ret_mix = mb_ret_mix.flatten()
        mb_actions = mb_actions.reshape(-1, *mb_actions.shape[2:])
        mb_v_ex = mb_v_ex.reshape(-1, *mb_v_ex.shape[2:])
        mb_v_mix = mb_v_mix.reshape(-1, *mb_v_mix.shape[2:])
        mb_dones = mb_dones.flatten()
    
        

        
        return mb_obs, mb_actions, mb_obs_nx, mb_states,mb_r_in, mb_r_ex, mb_ret_ex, mb_ret_mix, \
               mb_v_ex, mb_v_mix, last_v_ex, last_v_mix, mb_dones, \
               ep_info, ep_r_ex, ep_r_in, ep_len ,v_mix_terminal
        
