
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

OB_SPACE_SHAPE=181
STOCK_DIM=30 

class A2C_autoregressive_f(ActorCriticRLModel):
    """
    The A2C_autoregressive (Autoregressive Advantage Actor Critic) model class implemented using FNN 

    :param policy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) Discount factor
    :param n_steps: (int) The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param vf_coef: (float) Value function coefficient for the loss calculation
    :param ent_coef: (float) Entropy coefficient for the loss calculation
    :param max_grad_norm: (float) The maximum value for the gradient clipping
    :param learning_rate: (float) The learning rate
    :param lagstart: number of timesteps that policy starts updating before intrinsic 
    :param lagduring: number of timesteps per intrinsic update 

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
                 v_in_coef=0.5, v_ex_coef=1.0, r_in_coef=0.001,
                 ent_coef=0.01, max_grad_norm=0.5,
                 #learning_rate=7e-4,
                 lr_alpha = 7e-4, lr_beta = 7e-4,
                 alpha=0.99, momentum=0.0, epsilon=1e-5, lr_schedule='constant',preproc=False, avg_feat = None, std_feat=None ,
                 verbose=0, tensorboard_log=None,
                 _init_setup_model=True, policy_kwargs=None,
                 full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None, model_type=None ):
        print("MODEL: a2c_RNN") 
#use lr_alpha and lr_beta to replace learning_rate
        self.n_steps = n_steps
        self.gamma = gamma
        # avg_feature, std_feature are np.array of shape (181,) 
        
        
        if preproc: #True 
            avg_feature = avg_feat
            std_feature = std_feat 
        else: 
            avg_feature = np.zeros(OB_SPACE_SHAPE+30)
            std_feature = np.ones(OB_SPACE_SHAPE+30)
        
            
        self.avg_feature = np.reshape(avg_feature, (1, OB_SPACE_SHAPE+30)) 
        self.std_feature = np.reshape(std_feature,(1, OB_SPACE_SHAPE+ 30)) 
        
        self.train_model_avg_feature = np.tile(self.avg_feature, (self.n_steps*30, 1)) 
        self.train_model_std_feature = np.tile(self.std_feature, (self.n_steps*30 , 1)) 
        
        self.v_in_coef = v_in_coef
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
        self.DIS_V_IN_LAST = None
        self.V_IN = None
        self.A = None
        self.pg_mix_loss = None
        self.pg_ex_loss = None
        self.v_in_loss = None
        self.v_ex_loss = None
        self.entropy = None
        #self.apply_backprop = None
        self.train_model = None
        self.step_model = None
        self.proba_step = None
        self.value = None
        self.learning_rate_schedule = None
        self.summary = None

        ### super(): 1. Allows us to avoid using the base class name explicitly
        ### 2. Working with Multiple Inheritance
        super(A2C_autoregressive_f, self).__init__(policy=policy, env=env, verbose=verbose, requires_vec_env=True,
                                  _init_setup_model=_init_setup_model, policy_kwargs=policy_kwargs,
                                  seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)

        # if we are loading, it is possible the environment is not known, however the obs and action space are known
        if _init_setup_model:
            self.setup_model()

    def _make_runner(self) -> AbstractEnvRunner:
        return A2C_autoregressive_runner(self.env, self, n_steps=self.n_steps,r_ex_coef=self.r_ex_coef, r_in_coef=self.r_in_coef, gamma=self.gamma)

    def _get_pretrain_placeholders(self):  #not used
        policy = self.train_model
        if isinstance(self.action_space, spaces.Discrete):
            return policy.X, self.A, policy.policy
        return policy.X, self.A, policy.deterministic_action

    def setup_model(self):    # Part of the init in LIRPG A2C
        with SetVerbosity(self.verbose):
            
            self.graph = tf.Graph()
            with self.graph.as_default():
                self.set_random_seed(self.seed)
                self.sess = tf_util.make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)    # returns a session that will use <num_cpu> CPU's only
                self.n_batch = self.n_envs * self.n_steps

                n_batch_step = None
                n_batch_train = None
                if issubclass(self.policy, RecurrentActorCriticPolicy):
                    n_batch_step = self.n_envs
                    n_batch_train = self.n_envs * self.n_steps
                    
                step_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1, self.avg_feature, self.std_feature, reuse=False) 
                

                with tf.compat.v1.variable_scope("train_model", reuse=True,
                                       custom_getter=tf_util.outer_scope_getter("train_model")):
                    train_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs*self.n_steps*30, self.n_steps*30,self.train_model_avg_feature, self.train_model_std_feature,reuse=True)
                    
                    


                self.R_EX = tf.compat.v1.placeholder(tf.float32, [self.n_batch * 30], 'R_EX')
                self.RET_EX = tf.compat.v1.placeholder(tf.float32, [None], 'RET_EX')
                self.DIS_V_IN_LAST = tf.compat.v1.placeholder(tf.float32, [self.n_batch*30], 'DIS_V_IN_LAST')
                self.COEF_MAT = tf.compat.v1.placeholder(tf.float32, [self.n_batch*30, self.n_batch*30], 'COEF_MAT')
                self.V_IN = tf.compat.v1.placeholder(tf.float32, [self.n_batch *30], 'V_IN')
                self.V_EX = tf.compat.v1.placeholder(tf.float32, [self.n_batch *30], 'V_EX')
                

                self.A = tf.compat.v1.placeholder(tf.float32, [None, None], 'A') # used 
               
                
                r_in = train_model.r_in 
                ret_in = tf.squeeze(tf.matmul(self.COEF_MAT, tf.reshape(r_in, [self.n_batch*30, 1])), [1]) + self.DIS_V_IN_LAST
                ret_mix = self.r_ex_coef * self.RET_EX + self.r_in_coef * ret_in 
                
                
                
                adv_mix = ret_mix - (self.r_ex_coef * self.V_EX + self.r_in_coef * self.V_IN)  
                
                self.check = [r_in, self.R_EX, ret_in, ret_mix, self.RET_EX, adv_mix, self.V_IN, self.V_EX, self.r_ex_coef * self.V_EX + self.r_in_coef * self.V_IN]
                
                
                self.LR_ALPHA = tf.compat.v1.placeholder(tf.float32, [], name="LR_ALPHA")
                self.LR_BETA = tf.compat.v1.placeholder(tf.float32, [], name="LR_BETA")
                
                
                neglogpac = train_model.pd.neglogp(self.A)
            
                self.entropy = tf.reduce_mean(train_model.pd.entropy()) 
                
                
                
                self.pg_mix_loss = tf.reduce_mean(adv_mix * neglogpac)
                self.v_in_loss  = tf.reduce_mean(mse(tf.squeeze(train_model.v_mix), ret_in)) # train_model.v_mix = v_in 
                
                self.params = tf_util.get_trainable_vars("policy") # PI and V_IN parameters
                self.params1 = tf_util.get_trainable_vars("policy")[:7]  # PI only 

                policy_loss = self.pg_mix_loss - self.ent_coef * self.entropy + self.v_in_coef * self.v_in_loss
                policy_loss1= self.pg_mix_loss - self.ent_coef * self.entropy  

                with tf.compat.v1.variable_scope("policy_info", reuse=False): 
                    tf.compat.v1.summary.scalar('entropy_loss', self.entropy)
                    tf.compat.v1.summary.scalar('pg_mix_loss', self.pg_mix_loss)
                    tf.compat.v1.summary.scalar('v_mix_loss', self.v_in_loss)
                    tf.compat.v1.summary.scalar('policy_loss', policy_loss)
                    
                
                
                
                
                
                grads = tf.gradients(policy_loss, self.params)  
                grads1 = tf.gradients(policy_loss1, self.params1)  
                
                if self.max_grad_norm is not None:  # max_grad_norm defines the maximum gradient, needs to be normalized
                    # Clip the gradients (normalize)
                    grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
                    grads1, _ = tf.clip_by_global_norm(grads1, self.max_grad_norm)
                grads_and_vars = list(zip(grads, self.params))  # zip pg and policy params correspondingly, policy_grads_and_vars in LIRPG
                grads_and_vars1 = list(zip(grads1, self.params1))

                trainer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=self.LR_ALPHA, decay=self.alpha,
                                                    epsilon=self.epsilon, momentum=self.momentum) #Initialize optimizer 
                trainer1 = tf.compat.v1.train.RMSPropOptimizer(learning_rate=self.LR_ALPHA, decay=self.alpha, 
                                                    epsilon=self.epsilon, momentum=self.momentum) 


                self.policy_train = trainer.apply_gradients(grads_and_vars) 
                self.policy_train1 = trainer1.apply_gradients(grads_and_vars1) 
                
                rmss = [trainer.get_slot(var, 'rms') for var in self.params]   


                self.params_new = {}
                for grad, rms, var in zip(grads, rmss, self.params):
                    ms = rms + (tf.square(grad) - rms) * (1 - self.alpha) # wrong in this line
                    self.params_new[var.name] = var - self.LR_ALPHA * grad / tf.sqrt(ms + self.epsilon)  
                


                self.policy_new = None
                self.policy_new = train_model.policy_new_fn(self.params_new, self.observation_space, self.action_space, self.n_envs*self.n_steps*30, self.n_steps*30, self.train_model_avg_feature, self.train_model_std_feature)
            

                self.ADV_EX = tf.compat.v1.placeholder(tf.float32, [None], 'ADV_EX') #(n_steps,)
                
                   
                
                
                neglogpac_new = self.policy_new.pd.neglogp(self.A)
                
                ratio_new = tf.exp(tf.stop_gradient(neglogpac) - neglogpac_new)
                self.pg_ex_loss = tf.reduce_mean(-self.ADV_EX * ratio_new)
                self.v_ex_loss = tf.reduce_mean(mse(tf.squeeze(train_model.v_ex), self.RET_EX))
                intrinsic_loss = self.pg_ex_loss + self.v_ex_coef * self.v_ex_loss 
                
                
                
                self.intrinsic_params = tf_util.get_trainable_vars("intrinsic")   #A list of trainable variables [var1, var2, ....]
                
                
                self.intrinsic_vf_params = tf_util.get_trainable_vars("intrinsic")[6:]
                
                
                
                
                
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
                
                
                
                
               
                self.train_model = train_model
                
                
                
                
                self.step_model = step_model
                self.step = step_model.step 
                self.value = step_model.value
                self.intrinsic_reward =step_model.intrinsic_reward
                self.initial_state = step_model.initial_state
                
                tf.compat.v1.global_variables_initializer().run(session=self.sess)
                
        
                self.summary = tf.compat.v1.summary.merge_all()
            
        
    def _train_step(self,obs, actions,ob_nx, r_ex, ret_ex, v_ex, v_in, dis_v_in_last, coef_mat, update,writer=None,update_intrinsic=False ):


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
        
        
       
        
        
        td_map = {self.train_model.X: obs,  self.train_model.A_ALL: actions, self.train_model.X_NX:ob_nx,
                  self.policy_new.X:obs, 
                  self.A: actions, self.V_EX: v_ex,
                  self.ADV_EX: advs_ex, self.RET_EX:ret_ex, self.R_EX: r_ex, self.V_IN:v_in, self.DIS_V_IN_LAST:dis_v_in_last, self.COEF_MAT:coef_mat, 
                  self.LR_ALPHA:cur_lr_alpha, self.LR_BETA:cur_lr_beta} 
        
        
   
        
        if update_intrinsic:
            pg_ex_loss, pg_mix_loss, value_in_loss, value_ex_loss, policy_entropy, _, _ ,check_ = self.sess.run(
                    [self.pg_ex_loss, self.pg_mix_loss, self.v_in_loss, self.v_ex_loss, self.entropy, self.policy_train, self.intrinsic_train , self.check ], td_map) 
        
        
        
        else: 
            pg_mix_loss, value_in_loss,value_ex_loss, policy_entropy, _ , _, check_ = self.sess.run(
                    [self.pg_mix_loss, self.v_in_loss, self.v_ex_loss, self.entropy, self.policy_train, self.intrinsic_train1 , self.check], td_map) 
            pg_ex_loss =0
                
        
        return pg_ex_loss, pg_mix_loss, value_in_loss, value_ex_loss, policy_entropy , check_ 
    
            
            


    def learn(self, total_timesteps, start_intrinsic_update=50000, callback=None, log_interval=100, tb_log_name="A2C",
              reset_num_timesteps=True):
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
            
            
            if total_timesteps == 1600 : 
                self.mb_obs = self.runner.run_for_preproc() 
                return self 
            
        
            df= self.env.get_attr('df', indices=0)
            df_len = len(df[0].index.unique())
            total_timesteps = total_timesteps + (df_len - total_timesteps % df_len ) 
            print("total timesteps: ", total_timesteps ,'epochs' , total_timesteps/df_len )
            
            
            # A2C INTRINSIC TRAINING 

            for update in range(1, total_timesteps // self.n_batch + 1): 
                "env.day resets to 1 (by a2crunner) whenever a new instance of A2C is called. --> model doesnt start at day 256."
                
                if update % 1000 == 0: 
                    print("No. of days/ updates: ", update, "steps", update*30 )
                
                if update < start_intrinsic_update//self.n_batch: 
                    update_intrinsic=False 
                else: 
                    update_intrinsic = True 
                
                
                    
            
                #print("SELF.NUMTIMESTEPS: ", self.num_timesteps, "RINCOEF", r_in_coef, "REXCOEF", r_ex_coef)
                
                
                # Get mini batch of experiences
                callback.on_rollout_start()
                # true_reward is the reward without discount
                rollout = self.runner.run(callback)
                
                       
                obs, actions, ob_nx, r_in, r_ex, ret_ex, ret_in, \
                v_ex, v_in, last_v_ex, last_v_in, dones \
                ep_info, ep_r_ex, ep_r_in, ep_len = rollout 
                
              
                dis_v_in_last = np.zeros([self.n_batch*30], np.float32)
                coef_mat = np.zeros([self.n_batch*30, self.n_batch*30], np.float32)
                
                
                for i in range(self.n_batch*30):
                    dis_v_in_last[i] = self.gamma ** (self.n_steps*30 - i % (self.n_steps*30) )* last_v_in[0]

                    coef = 1.0
                    for j in range(i, self.n_batch * 30):
                        if j > i and j % (self.n_steps*30) == 0:
                            break
                        coef_mat[i][j] = coef
                        coef *= self.gamma 
                        if dones[j]:
                            dis_v_in_last[i] = 0
                            break



                        
                        
                callback.update_locals(locals())
                callback.on_rollout_end()

                # Early stopping due to the callback
                if not self.runner.continue_training:
                    break

                self.ep_info_buf.extend(ep_info)
                self.eprexbuf.extend(ep_r_ex)
                self.eplenbuf.extend(ep_len)

             
                
                pg_ex_loss, pg_mix_loss, value_in_loss, value_ex_loss, policy_entropy , check_ = self._train_step(obs, actions,ob_nx, r_ex, ret_ex, v_ex, v_in,
                                                                 dis_v_in_last, coef_mat, self.num_timesteps // self.n_batch, writer,update_intrinsic)
                
                
                
                
                policy_loss = pg_mix_loss - self.ent_coef * policy_entropy + self.v_in_coef * value_in_loss
                
                intrinsic_loss = pg_ex_loss + self.v_ex_coef * value_ex_loss
                
               
            

            
                n_seconds = time.time() - t_start
                # Calculate the fps (frame per second)
                fps = int((update * self.n_batch) / n_seconds)

                
        callback.on_training_end()
        return self

    def predict_intrinsic(self, observation, state=None, mask=None, deterministic=False): 
        action_ph = np.zeros(shape=(1,30)) # Reset action placeholder before starting actions 1-30 
        
        for j in range(30): 
            input_ = np.concatenate(( np.copy(observation) , action_ph), axis=1 ).reshape((1,211)) # (1,182)
            new_action, _, _, _, _ = self.step(input_ )
            new_action = np.clip(new_action,  -1, 1) # (1,1)
            action_ph[0,j] = new_action
        return action_ph 


    def save(self, save_path, cloudpickle=False):
        data = {
            "gamma": self.gamma,
            "n_steps": self.n_steps,
            #"vf_coef": self.vf_coef,
            "v_mix_coef": self.v_in_coef,
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





class A2C_autoregressive_runner(AbstractEnvRunner):
    def __init__(self, env, model, n_steps=5,r_ex_coef=1-0.001,nlstm =30, r_in_coef=0.001, gamma=0.99):
        """
        A runner to learn the policy of an environment for an a2c model

        :param env: (Gym environment) The environment to learn from
        :param model: (Model) The model to learn
        :param n_steps: (int) The number of steps to run for each environment
        :param gamma: (float) Discount factor
        """
        #line 162-176: needs to modify the parameters in runner; MARKED LINES TO TAKE CARE LATER
        super(A2C_autoregressive_runner, self).__init__(env=env, model=model, n_steps=n_steps)
        self.gamma = gamma

   
        nenv = env.num_envs
        self.r_ex_coef = r_ex_coef
        self.r_in_coef = r_in_coef
        self.ep_r_in = np.zeros([nenv])
        self.ep_r_ex = np.zeros([nenv])
        self.ep_len = np.zeros([nenv])
        
    def run_for_preproc(self): # Start of state. 
        mb_obs=[] 
        action_ph = np.zeros(shape=(1,30))
        
        
        self.internal_count = 0
        for i in range(self.n_steps*30): 
            
            input_ = np.concatenate(( np.copy(self.obs) , action_ph), axis=1 ).reshape((1,OB_SPACE_SHAPE+30)) # (1,182)
            
            new_action, _, _, _, _ = self.model.step(input_ )
            mb_obs.append(np.copy(input_))
            
            
            new_action = np.clip(new_action,  -1, 1) # (1,1)
            
            action_ph[0,self.internal_count] = new_action 
            
            self.internal_count += 1  # NO OF ACTIONS IN STATE SO FAR. 
            
            if self.internal_count == 30: 
                self.obs, r_ex, self.dones, infos = self.env.step(action_ph)
                action_ph = np.zeros(shape=(1,30))
                self.internal_count = 0 
           
             
       
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype).swapaxes(1, 0).reshape( (30*self.n_steps, OB_SPACE_SHAPE+30 )) # (30, 211)
        
        
        return mb_obs

    
        

    def _run(self):

        mb_obs, mb_r_ex, mb_r_in, mb_actions, mb_v_ex, mb_v_in= [],[],[],[],[],[]
        mb_obs_next = []
        ep_info, ep_r_ex, ep_r_in, ep_len = [], [], [], []
        
        
        i=0 
        while i < self.n_steps * STOCK_DIM: 

            action_ph = np.zeros(shape=(1,30)) # Reset action placeholder before starting actions 1-30 
            
            for j in range(STOCK_DIM): 
                input_ = np.concatenate(( np.copy(self.obs) , action_ph), axis=1 ).reshape((1,OB_SPACE_SHAPE+30)) # (1,182)
                
                new_action, v_ex, v_in, _, _ = self.model.step(input_ )
                
                mb_obs.append(np.copy(input_))
                mb_actions.append(new_action) 
                
                new_action = np.clip(new_action,  -1, 1) 
                
                action_ph[0,j] = new_action 
                
                mb_v_in.append(v_in) 
                mb_v_ex.append(v_ex)
                
               
                if j < STOCK_DIM-1:
                    r_ex = np.array([0])
                    mb_r_ex.append(r_ex) 
                    next_input_ = np.concatenate(( np.copy(self.obs) , action_ph), axis=1 ).reshape((1,OB_SPACE_SHAPE+30))
                    r_in = self.model.intrinsic_reward(ob= input_ , ac=new_action , ob_nx=next_input_ ) 
                    mb_r_in.append(r_in)
                    mb_obs_next.append(next_input_ )
                    
                elif j == STOCK_DIM - 1:    
                    old_obs =np.copy( self.obs)                  
                    self.obs, r_ex, self.dones, infos = self.env.step(action_ph)
                    if self.dones: 
                        r_ex= np.array([0]) 
                        mb_r_ex.append(r_ex) 
                        next_input_ = np.concatenate(( old_obs , action_ph), axis=1 ).reshape((1,OB_SPACE_SHAPE+30))
                        mb_obs_next.append(next_input_ )
                        r_in = self.model.intrinsic_reward(ob= input_ , ac=new_action , ob_nx=next_input_ ) 
                        mb_r_in.append(r_in)
                        last_v_ex, last_v_in = self.model.value(ob =next_input_ ) 
                    
                    else: 
                        mb_r_ex.append(r_ex) 
                        action_ph = np.zeros(shape=(1,30))
                        next_input_ = np.concatenate(( np.copy(self.obs) , action_ph), axis=1 ).reshape((1,OB_SPACE_SHAPE+30))
                        mb_obs_next.append(next_input_ )
                        r_in = self.model.intrinsic_reward(ob= input_ , ac=new_action , ob_nx=next_input_ ) 
                        mb_r_in.append(r_in)
                        last_v_ex, last_v_in = self.model.value(ob =next_input_ ) 
                
                    i+= STOCK_DIM 
                    
                    self.model.num_timesteps += self.n_envs 
    
       
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype).swapaxes(1, 0).reshape( (self.n_steps*STOCK_DIM, OB_SPACE_SHAPE+30 )) 
        mb_obs_nx = np.asarray(mb_obs_next, dtype=self.obs.dtype).swapaxes(1, 0).reshape((self.n_steps*STOCK_DIM , OB_SPACE_SHAPE+30))
        
        mb_r_ex = np.asarray(mb_r_ex, dtype=np.float32).swapaxes(1, 0) 
        mb_r_in = np.asarray(mb_r_in, dtype=np.float32).swapaxes(1, 0)
        
        
        mb_actions = np.asarray(mb_actions, dtype=self.env.action_space.dtype).swapaxes(0, 1) # ( 30, 1)
        mb_v_ex = np.asarray(mb_v_ex, dtype=np.float32).swapaxes(1, 0) 
        
        
        
        mb_v_in = np.asarray(mb_v_in, dtype=np.float32).swapaxes(1, 0)
    
                        
        mb_ret_ex, mb_ret_in = np.zeros(mb_r_ex.shape), np.zeros(mb_r_in.shape)
        
        # discount/bootstrap off value fn
            
            
        
        for n, (r_ex, r_in, v_ex, v_in) in enumerate(zip(mb_r_ex, mb_r_in, last_v_ex, last_v_in)): 
            r_ex, r_in = r_ex.tolist(), r_in.tolist() 
            dones = [False] * 29 + [ self.dones[0] ]  
            if dones[-1] == 0:   # last state not terminal        
                ret_ex = discount_with_dones(r_ex + [0.99*v_ex], dones + [0], 1)[:-1] 
                ret_in = discount_with_dones(r_in+ [v_in ], dones + [0], self.gamma )[:-1]   # Ret_in 
            else: # last state terminal. 
                ret_ex = discount_with_dones(r_ex, dones, 1 )
                ret_in = discount_with_dones(r_in, dones, self.gamma)
            
            #mb_rewards[n] = rewards
            mb_ret_ex[n], mb_ret_in[n] = ret_ex, ret_in 


            
       
        mb_dones = [False] * 29 + [ self.dones[0] ]   
        mb_dones = np.asarray(mb_dones, dtype=np.bool_) # (30,) array 
        mb_r_ex = mb_r_ex.flatten()
        mb_r_in = mb_r_in.flatten()
        mb_ret_ex = mb_ret_ex.flatten()
        mb_ret_in = mb_ret_in.flatten()
        mb_actions = mb_actions.reshape(-1, *mb_actions.shape[2:])
        mb_v_ex = mb_v_ex.reshape(-1, *mb_v_ex.shape[2:])
        mb_v_in = mb_v_in.reshape(-1, *mb_v_in.shape[2:])
    
        
        return mb_obs, mb_actions,mb_obs_nx,mb_r_in, mb_r_ex, mb_ret_ex, mb_ret_in, \
               mb_v_ex, mb_v_in, last_v_ex, last_v_in, mb_dones, \
               ep_info, ep_r_ex, ep_r_in, ep_len 
        