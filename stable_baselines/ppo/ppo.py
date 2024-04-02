

OB_SPACE_SHAPE=181 
STOCK_DIM = 30 



import time
import sys
import csv                                                                                                                                                                                                                                                                                                                                                                                    


import gym 
from gym import spaces
from gym.utils import seeding


import tensorflow as tf
import numpy as np 

from stable_baselines import logger
from stable_baselines.common import explained_variance, tf_util, ActorCriticRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.policies import ActorCriticPolicy, RecurrentActorCriticPolicy
from stable_baselines.common.runners import AbstractEnvRunner
from stable_baselines.common.schedules import Scheduler, PiecewiseSchedule 
from stable_baselines.common.tf_util import mse, total_episode_reward_logger, calc_entropy
from stable_baselines.common.math_util import safe_mean

tf.compat.v1.reset_default_graph()  # I ADDED . 


class PPO(ActorCriticRLModel): 
    def __init__(self, policy, env, gamma=0.99, n_steps=30, #vf_coef=0.25,
                 v_mix_coef=0.5, v_ex_coef=1.0, r_in_coef=0.001, nminibatches=6, noptepochs=1, cliprange=0.2, 
                 ent_coef=0.01, max_grad_norm=0.5,lam=0.95, reward_freq=1, 
                 #learning_rate=7e-4,
                 lr_alpha = 7e-4, lr_beta = 7e-4,
                 alpha=0.99, momentum=0.0, epsilon=1e-5, lr_schedule='constant',preproc=False, avg_feat = None, std_feat=None ,
                 verbose=0, tensorboard_log=None, 
                 _init_setup_model=True, policy_kwargs=None,
                 full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None, model_type=None ):
        if r_in_coef ==0: 
            print("MODEL: Non-Autoregressive PPO (no intrinsic)")
        else: 
            print("MODEL: Non-Autoregressive PPO + Intrinsic")
        self.n_steps = n_steps 
        
        self.gamma = gamma 
        self.lam=lam 
        self.nminibatches= nminibatches 
        self.noptepochs = noptepochs  
        self.cliprange= cliprange 
        self.reward_freq = reward_freq
        
        
        if preproc: 
            avg_feature = avg_feat
            std_feature = std_feat 
        else: 
            avg_feature = np.zeros(OB_SPACE_SHAPE)
            std_feature = np.ones(OB_SPACE_SHAPE)
        
        
        self.avg_feature = np.reshape(avg_feature, (1, OB_SPACE_SHAPE)) 
        self.std_feature = np.reshape(std_feature,(1, OB_SPACE_SHAPE)) 
        
       
        self.train_model_avg_feature = np.tile(self.avg_feature, (self.n_steps // self.nminibatches, 1)) 
        self.train_model_std_feature = np.tile(self.std_feature, (self.n_steps // self.nminibatches, 1)) 
        
        
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
        self.lr_alpha = lr_alpha
        self.lr_beta = lr_beta
        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log

        self.LR_ALPHA = None
        self.LR_BETA = None
        self.ADV_EX = None
        self.RET_EX = None
        self.R_EX = None
        self.DIS_V_MIX_LAST = None
        self.V_MIX = None
        self.A = None
        
        self.pg_mix_loss = None
        self.pg_ex_loss = None
        self.v_mix_loss = None
        self.v_ex_loss = None
        self.entropy = None
        self.train_model = None
        self.step_model = None
        self.proba_step = None
        self.value = None
        self.learning_rate_schedule = None
        self.summary = None

       
        super(PPO, self).__init__(policy=policy, env=env, verbose=verbose, requires_vec_env=True,
                                  _init_setup_model=_init_setup_model, policy_kwargs=policy_kwargs,
                                  seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)

        if _init_setup_model:
            self.setup_model()

    def _make_runner(self) -> AbstractEnvRunner:
        return PPO_Runner(self.env, self, nsteps=self.n_steps, gamma=self.gamma, lam=self.lam , r_ex_coef=self.r_ex_coef, r_in_coef=self.r_in_coef,reward_freq=self.reward_freq )

    def _get_pretrain_placeholders(self):  #not used
        policy = self.train_model
        if isinstance(self.action_space, spaces.Discrete):
            return policy.X, self.A, policy.policy
        return policy.X, self.A, policy.deterministic_action

    def setup_model(self):   
        with SetVerbosity(self.verbose):
           
            self.graph = tf.Graph()
            with self.graph.as_default():
                self.set_random_seed(self.seed)
                self.nbatch = self.n_envs * self.n_steps 
                self.sess = tf_util.make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)    
                
                
                assert self.nbatch % self.nminibatches == 0 
                
                self.nbatch_train = self.nbatch // self.nminibatches 
                

                n_batch_step = None
                n_batch_train = None
                if issubclass(self.policy, RecurrentActorCriticPolicy):
                    n_batch_step = self.n_envs
                    n_batch_train = self.n_envs * self.n_steps
                    
                step_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1, self.avg_feature, self.std_feature, reuse=False)  
                
              

                with tf.compat.v1.variable_scope("train_model", reuse=True,
                                       custom_getter=tf_util.outer_scope_getter("train_model")):
                    train_model = self.policy(self.sess, self.observation_space, self.action_space, self.nbatch_train , self.n_steps,self.train_model_avg_feature, self.train_model_std_feature,reuse=True)
                    

                

                self.A = train_model.pdtype.sample_placeholder([self.nbatch_train], "A")
                self.OLDNEGLOGPAC = tf.compat.v1.placeholder(tf.float32, [self.nbatch_train], "OLDNEGLOGPAC")
                self.R_EX = tf.compat.v1.placeholder(tf.float32, [self.nbatch], "R_EX")
                self.ADV_EX = tf.compat.v1.placeholder(tf.float32, [self.nbatch_train], "ADV_EX") 
                self.RET_EX = tf.compat.v1.placeholder(tf.float32, [self.nbatch_train], "RET_EX")
                self.OLDV_EX = tf.compat.v1.placeholder(tf.float32, [self.nbatch_train], "OLDV_EX")
                self.OLDV_MIX = tf.compat.v1.placeholder(tf.float32, [self.nbatch_train], "OLDV_MIX")
                self.TD_MIX = tf.compat.v1.placeholder(tf.float32, [self.nbatch], "TD_MIX")
                self.COEF_MAT = tf.compat.v1.placeholder(tf.float32, [self.nbatch_train, self.nbatch], "COEF_MAT")
                self.CLIPRANGE = tf.compat.v1.placeholder(tf.float32, [])
                self.LR_ALPHA = tf.compat.v1.placeholder(tf.float32, [], "LR_ALPHA")
                self.LR_BETA = tf.compat.v1.placeholder(tf.float32, [], "LR_BETA")

                
                
                # SIMULATE adv mix: 
                delta_mix = self.r_in_coef * train_model.r_in + self.r_ex_coef * self.R_EX + self.TD_MIX
                adv_mix = tf.squeeze(tf.matmul(self.COEF_MAT, tf.reshape(delta_mix, [self.nbatch, 1])), [1])
                ret_mix = adv_mix + self.OLDV_MIX
                adv_mix_mean, adv_mix_var = tf.compat.v1.nn.moments(adv_mix, axes=0)
                adv_mix = (adv_mix - adv_mix_mean) / (tf.sqrt(adv_mix_var) + 1E-8)
            
                neglogpac = train_model.pd.neglogp(self.A)
                self.entropy = tf.reduce_mean(train_model.pd.entropy())

        
               
                # CALCULATE policy loss
                
                ratio = tf.compat.v1.exp(self.OLDNEGLOGPAC - neglogpac)
                pg_mix_loss1 = -adv_mix * ratio
                pg_mix_loss2 = -adv_mix * tf.compat.v1.clip_by_value(ratio, 1.0 - self.CLIPRANGE, 1.0 + self.CLIPRANGE)
                self.pg_mix_loss = tf.reduce_mean(tf.maximum(pg_mix_loss1, pg_mix_loss2))
                self.approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - self.OLDNEGLOGPAC))
                self.clipfrac = tf.reduce_mean(tf.compat.v1.to_float(tf.greater(tf.abs(ratio- 1.0), self.CLIPRANGE)))
                v_mix = train_model.v_mix
                v_mix_clipped = self.OLDV_MIX + tf.compat.v1.clip_by_value(v_mix - self.OLDV_MIX, - self.CLIPRANGE, self.CLIPRANGE)
                v_mix_loss1 = tf.square(v_mix - ret_mix)
                v_mix_loss2 = tf.square(v_mix_clipped - ret_mix)
                self.v_mix_loss = .5 * tf.reduce_mean(tf.maximum(v_mix_loss1, v_mix_loss2))
                policy_loss = self.pg_mix_loss - self.entropy * self.ent_coef + self.v_mix_loss * self.v_mix_coef
                

                with tf.compat.v1.variable_scope("policy_info", reuse=False): 
                    tf.compat.v1.summary.scalar('entropy_loss', self.entropy)
                    tf.compat.v1.summary.scalar('pg_mix_loss', self.pg_mix_loss)
                    tf.compat.v1.summary.scalar('v_mix_loss', self.v_mix_loss)
                    tf.compat.v1.summary.scalar('policy_loss', policy_loss)
                    
                self.params = tf_util.get_trainable_vars("policy")  
                grads = tf.gradients(policy_loss, self.params)   
                if self.max_grad_norm is not None: 
                    grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
                grads_and_vars = list(zip(grads, self.params)) 
                
                trainer= tf.compat.v1.train.AdamOptimizer(learning_rate=self.LR_ALPHA, epsilon=self.epsilon)
                
                
                
                self.policy_train = trainer.apply_gradients(grads_and_vars)  
                beta1_power, beta2_power = trainer._get_beta_accumulators()
                

                self.params_new = {}
                for var, grad in zip(self.params, grads):
                    lr_ = self.LR_ALPHA * tf.sqrt(1 - beta2_power) / (1 - beta1_power)
                    m, v = trainer.get_slot(var, 'm'), trainer.get_slot(var, 'v')
                    m = m + (grad - m) * (1 - .9)
                    v = v + (tf.square(tf.stop_gradient(grad)) - v) * (1 - .999)
                    self.params_new[var.name] = var - m * lr_ / (tf.sqrt(v) + 1E-5)
                
                self.policy_new = None
                self.policy_new = train_model.policy_new_fn(self.params_new, self.observation_space, self.action_space, self.nbatch_train, self.n_steps, self.train_model_avg_feature, self.train_model_std_feature)
            
            
            
                neglogpac_new = self.policy_new.pd.neglogp(self.A)
                ratio_new = tf.exp(self.OLDNEGLOGPAC - neglogpac_new)
                pg_ex_loss1 = -self.ADV_EX * ratio_new
                pg_ex_loss2 = -self.ADV_EX * tf.clip_by_value(ratio_new, 1.0 - self.CLIPRANGE, 1.0 + self.CLIPRANGE)
                self.pg_ex_loss = tf.reduce_mean(tf.maximum(pg_ex_loss1, pg_ex_loss2))
                v_ex = train_model.v_ex
                v_ex_clipped = self.OLDV_EX + tf.clip_by_value(v_ex - self.OLDV_EX, - self.CLIPRANGE, self.CLIPRANGE)
                v_ex_loss1 = tf.square(v_ex - self.RET_EX)
                v_ex_loss2 = tf.square(v_ex_clipped - self.RET_EX)
                self.v_ex_loss = .5 * tf.reduce_mean(tf.maximum(v_ex_loss1, v_ex_loss2))
                intrinsic_loss = self.pg_ex_loss + self.v_ex_coef * self.v_ex_loss 
                
                
                self.intrinsic_params = tf_util.get_trainable_vars("intrinsic")  
                
                self.intrinsic_vf_params = tf_util.get_trainable_vars("intrinsic")[0:6]
                
                
            
 
            
                intrinsic_grads = tf.gradients(intrinsic_loss, self.intrinsic_params)
                
                intrinsic_grads1 = tf.gradients(self.v_ex_loss, self.intrinsic_vf_params)
                

                if self.max_grad_norm is not None:
                    
                    intrinsic_grads, _ = tf.clip_by_global_norm(intrinsic_grads, self.max_grad_norm)
                    intrinsic_grads1, _ = tf.clip_by_global_norm(intrinsic_grads1, self.max_grad_norm)
                                        
                intrinsic_grads_and_vars = list(zip(intrinsic_grads, self.intrinsic_params))
                intrinsic_grads_and_vars1 = list(zip(intrinsic_grads1, self.intrinsic_vf_params))
                

                intrinsic_trainer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.LR_BETA, epsilon=self.epsilon)
                
                intrinsic_trainer1 =tf.compat.v1.train.AdamOptimizer(learning_rate=self.LR_BETA, epsilon=self.epsilon)
                
                

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
            
    def _train_step(self, obs, obs_all, obs_nx_all, actions, actions_all, neglogpacs, states, masks, r_ex, ret_ex, v_ex, td_mix, v_mix, coef_mat,update_intrinsic=True,writer=None ):   
    
        adv_ex = ret_ex - v_ex
        adv_ex = (adv_ex - adv_ex.mean()) / (adv_ex.std() + 1e-8) 
        for _ in range(len(obs)):
            cur_lr_alpha = self.lr_alpha.value()            
            cur_lr_beta = self.lr_beta.value()
        assert cur_lr_alpha is not None, "Error: the observation input array cannon be empty"
        assert cur_lr_beta is not None, "Error: the observation input array cannon be empty"
        
        td_map = {self.train_model.X:obs, self.train_model.X_ALL:obs_all, self.policy_new.X:obs, self.train_model.X_NX: obs_nx_all, 
                      self.A:actions, self.train_model.A_ALL:actions_all, self.OLDNEGLOGPAC:neglogpacs,
                      self.R_EX:r_ex, self.ADV_EX:adv_ex, self.RET_EX:ret_ex, self.OLDV_EX:v_ex, self.OLDV_MIX:v_mix, self.TD_MIX:td_mix,
                      self.COEF_MAT:coef_mat, self.CLIPRANGE:self.cliprange,self.LR_ALPHA:cur_lr_alpha, self.LR_BETA:cur_lr_beta}
        
       
        
       
        if states is not None:
            td_map[self.train_model.S] = states

       
        
        if update_intrinsic:
            pg_ex_loss, pg_mix_loss, value_mix_loss, value_ex_loss, policy_entropy,approxkl_, clipfrac_, _, _  = self.sess.run(
                    [self.pg_ex_loss, self.pg_mix_loss, self.v_mix_loss, self.v_ex_loss, self.entropy,self.approxkl, self.clipfrac, self.policy_train, self.intrinsic_train ], td_map) 
        
        else: 
            pg_mix_loss, value_mix_loss,value_ex_loss, policy_entropy,approxkl_, clipfrac_, _ , _ = self.sess.run(
                    [self.pg_mix_loss, self.v_mix_loss, self.v_ex_loss, self.entropy,self.approxkl, self.clipfrac, self.policy_train, self.intrinsic_train1 ], td_map) 
            
            pg_ex_loss =0
                
        
        return pg_ex_loss, pg_mix_loss, value_mix_loss, value_ex_loss, policy_entropy ,approxkl_, clipfrac_
    
            
            


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
            
           
            
            # A2C INTRINSIC TRAINING 

            for update in range(1, total_timesteps // self.nbatch + 1): 
                
                if update % 1000 == 0: 
                    print("No. of days/ updates: ", update, "steps", update*30 )
                
                
                if update < start_intrinsic_update//self.nbatch: 
                    update_intrinsic=False 
                else: 
                    update_intrinsic = True 
                
                
                
                # Get mini batch of experiences
                callback.on_rollout_start()
                rollout = self.runner.run(callback)
                
                obs, masks, actions, neglogpacs, r_ex, r_in, ret_ex, ret_mix, v_ex, v_mix, td_mix, mb_obs_nx , states, \
                epinfos, ep_r_ex, ep_r_in, ep_len  = rollout 
                       
                      


                
                if total_timesteps == 1600: 
                    self.mb_obs = obs 
                    return self 
                
                if states is None: # nonrecurrent version
                    inds = np.arange(self.nbatch) 
                    for _ in range(self.noptepochs): # for noptepochs epochs 
                        np.random.shuffle(inds) # indices are shuffled for each epoch. 
                        for start in range(0, self.nbatch, self.nbatch_train): # each minibatch in each epoch 
                            end = start + self.nbatch_train
                            mbinds = inds[start:end]
                            coef_mat = np.zeros([self.nbatch_train, self.nbatch], "float32") # 4, 20 
                            for i in range(self.nbatch_train): # 4 
                                coef = 1.0
                                for j in range(mbinds[i], self.nbatch): # 20 
                                    if j > mbinds[i] and (masks[j] or j % self.n_steps == 0):
                                        break
                                    coef_mat[i][j] = coef
                                    coef *= self.gamma * self.lam
                                    
                            pg_ex_loss, pg_mix_loss, value_mix_loss, value_ex_loss, policy_entropy,approxkl, clipfrac = self._train_step(obs[mbinds], obs,mb_obs_nx, actions[mbinds], actions, neglogpacs[mbinds],\
                                        None, masks[mbinds], r_ex, ret_ex[mbinds], v_ex[mbinds], td_mix,\
                                        v_mix[mbinds], coef_mat, update_intrinsic )
                                
                       
                                
                
                        
                callback.update_locals(locals())
                callback.on_rollout_end()

                # Early stopping due to the callback
                if not self.runner.continue_training:
                    break

                self.ep_info_buf.extend(epinfos)
                self.eprexbuf.extend(ep_r_ex)
                self.eplenbuf.extend(ep_len)
                
                
                
                
                
                
                

 
                n_seconds = time.time() - t_start
                
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





class PPO_Runner(AbstractEnvRunner):
    def __init__(self, env, model, nsteps, gamma,lam, r_ex_coef, r_in_coef, reward_freq):
        """
        A runner to learn the policy of an environment for an a2c model

        :param env: (Gym environment) The environment to learn from
        :param model: (Model) The model to learn
        :param n_steps: (int) The number of steps to run for each environment
        :param gamma: (float) Discount factor
        """
        #line 162-176: needs to modify the parameters in runner; MARKED LINES TO TAKE CARE LATER
        super(PPO_Runner, self).__init__(env=env, model=model, n_steps = nsteps )
        self.gamma = gamma
        self.lam = lam 

    
        nenv = env.num_envs
        self.r_ex_coef = r_ex_coef
        self.r_in_coef = r_in_coef
        self.ep_r_in = np.zeros([nenv])
        self.ep_r_ex = np.zeros([nenv])
        self.ep_len = np.zeros([nenv])
        self.states = model.initial_state
        self.reward_freq= reward_freq 
        self.delay_r_ex = np.zeros([nenv])
        self.delay_step = np.zeros([nenv])
        
        
        
        
    def _run(self):

                
        mb_obs, mb_r_ex, mb_r_in, mb_actions, mb_v_ex, mb_v_mix, mb_dones, mb_neglogpacs = [],[],[],[],[],[],[],[]
        mb_obs_next = []
        mb_states = self.states
        #ep_infos = []
        ep_info, ep_r_ex, ep_r_in, ep_len = [], [], [], []
        for _ in range(self.n_steps): 
  
            
            
            actions, v_ex, v_mix, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)
            
            mb_obs.append(np.copy(self.obs))
            mb_v_ex.append(v_ex)
            mb_v_mix.append(v_mix)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            clipped_actions = np.clip(actions,  self.env.action_space.low, self.env.action_space.high)
            mb_actions.append(clipped_actions)
            
            
            obs, r_ex, self.dones, infos = self.env.step(clipped_actions) 
            
            self.delay_r_ex += r_ex
            self.delay_step += 1
            for n, done in enumerate(self.dones):
                if done or self.delay_step[n] == self.reward_freq:
                    r_ex[n] = self.delay_r_ex[n]
                    self.delay_r_ex[n] = self.delay_step[n] = 0
                else:
                    r_ex[n] = 0
                    
            mb_r_ex.append(r_ex) 
            mb_obs_next.append(obs) 
            r_in = self.model.intrinsic_reward(ob=self.obs, ac=clipped_actions, ob_nx= obs)    ###EXPAND TO INCULDE OBS_NX, obs = OBS_NX, self.obs = OBS
            mb_r_in.append(r_in)
            for n, done in enumerate(self.dones):
                if done:
                    self.obs[n] = self.obs[n]*0
                    ep_r_ex.append(self.ep_r_ex[n])
                    ep_r_in.append(self.ep_r_in[n])
                    ep_len.append(self.ep_len[n])
                    self.ep_r_ex[n], self.ep_r_in[n], self.ep_len[n] = 0,0,0
            
            
            

            self.ep_r_ex += r_ex 
            self.ep_r_in += r_in
            self.ep_len += 1



            self.obs =obs 
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        
      
        mb_actions = np.asarray(mb_actions, dtype=self.env.action_space.dtype)
        
        mb_obs_nx = np.asarray(mb_obs_next, dtype=self.obs.dtype)
        
        
        
        
        mb_r_ex = np.asarray(mb_r_ex, dtype=np.float32)
        mb_r_in = np.asarray(mb_r_in, dtype=np.float32)
        mb_r_mix = self.r_ex_coef * mb_r_ex + self.r_in_coef * mb_r_in
        
        
        
        
        
        mb_v_ex = np.asarray(mb_v_ex, dtype=np.float32)
        
        
        
        mb_v_mix = np.asarray(mb_v_mix, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
                
        mb_dones = np.asarray(mb_dones, dtype=np.bool_)
        
        
      
        
        last_v_ex, last_v_mix = self.model.value(self.obs, self.states, self.dones )
        mb_v_mix_next = np.zeros_like(mb_v_mix)
        mb_v_mix_next[:-1] = mb_v_mix[1:] * (1.0 - mb_dones[1:])
        mb_v_mix_next[-1] = last_v_mix * (1.0 - self.dones)
        td_mix = self.gamma * mb_v_mix_next - mb_v_mix
        mb_adv_ex = np.zeros_like(mb_r_ex)
        mb_adv_mix = np.zeros_like(mb_r_mix)
        lastgaelam_ex, lastgaelam_mix = 0,0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                nextnonterminal = 1.0 - self.dones
                nextv_ex = last_v_ex
                nextv_mix = last_v_mix
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextv_ex = mb_v_ex[t+1]
                nextv_mix = mb_v_mix[t+1]
            delta_ex = mb_r_ex[t] + self.gamma * nextv_ex * nextnonterminal - mb_v_ex[t]
            delta_mix = mb_r_mix[t] + self.gamma * nextv_mix * nextnonterminal - mb_v_mix[t]
            mb_adv_ex[t] = lastgaelam_ex = delta_ex + self.gamma * self.lam * nextnonterminal * lastgaelam_ex
            mb_adv_mix[t] = lastgaelam_mix = delta_mix + self.gamma * self.lam * nextnonterminal * lastgaelam_mix
        mb_ret_ex = mb_adv_ex + mb_v_ex
        mb_ret_mix = mb_adv_mix + mb_v_mix
                        
        
            
        return (*map(sf01, (mb_obs, mb_dones, mb_actions, mb_neglogpacs,
                            mb_r_ex, mb_r_in, mb_ret_ex, mb_ret_mix, mb_v_ex, mb_v_mix, td_mix, mb_obs_nx)),
            mb_states, ep_info, ep_r_ex, ep_r_in, ep_len)   

        
                        
       

        
    
        
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])



