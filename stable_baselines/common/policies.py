import warnings
from itertools import zip_longest
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from gym.spaces import Discrete

from stable_baselines.common.tf_util import batch_to_seq, seq_to_batch
from stable_baselines.common.tf_layers import conv, linear, conv_to_fc, fc
from stable_baselines.common.distributions import make_proba_dist_type, CategoricalProbabilityDistribution, \
    MultiCategoricalProbabilityDistribution, DiagGaussianProbabilityDistribution, BernoulliProbabilityDistribution, DiagGaussianProbabilityDistributionType
from stable_baselines.common.input import observation_input



AVG_FEATURE = None #th.zeros(181)
STDEV_FEATURE = None #th.ones(181)
K_MAX = 1.

def nature_cnn(scaled_images, **kwargs):
    """
    CNN from Nature paper.

    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    activ = tf.nn.relu
    #print(scaled_images)
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = conv_to_fc(layer_3)
    return activ(linear(layer_3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))


def mlp_extractor(flat_observations, net_arch, act_fun):
    """
    Constructs an MLP that receives observations as an input and outputs a latent representation for the policy and
    a value network. The ``net_arch`` parameter allows to specify the amount and size of the hidden layers and how many
    of them are shared between the policy network and the value network. It is assumed to be a list with the following
    structure:

    1. An arbitrary length (zero allowed) number of integers each specifying the number of units in a shared layer.
       If the number of ints is zero, there will be no shared layers.
    2. An optional dict, to specify the following non-shared layers for the value network and the policy network.
       It is formatted like ``dict(vf=[<value layer sizes>], pi=[<policy layer sizes>])``.
       If it is missing any of the keys (pi or vf), no non-shared layers (empty list) is assumed.

    For example to construct a network with one shared layer of size 55 followed by two non-shared layers for the value
    network of size 255 and a single non-shared layer of size 128 for the policy network, the following layers_spec
    would be used: ``[55, dict(vf=[255, 255], pi=[128])]``. A simple shared network topology with two layers of size 128
    would be specified as [128, 128].

    :param flat_observations: (tf.Tensor) The observations to base policy and value function on.
    :param net_arch: ([int or dict]) The specification of the policy and value networks.
        See above for details on its formatting.
    :param act_fun: (tf function) The activation function to use for the networks.
    :return: (tf.Tensor, tf.Tensor) latent_policy, latent_value of the specified network.
        If all layers are shared, then ``latent_policy == latent_value``
    """
    latent = flat_observations
    policy_only_layers = []  # Layer sizes of the network that only belongs to the policy network
    value_only_layers = []  # Layer sizes of the network that only belongs to the value network

    # Iterate through the shared layers and build the shared parts of the network
    for idx, layer in enumerate(net_arch):
        if isinstance(layer, int):  # Check that this is a shared layer
            layer_size = layer
            latent = act_fun(linear(latent, "shared_fc{}".format(idx), layer_size, init_scale=np.sqrt(2)))
        else:
            assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
            if 'pi' in layer:
                assert isinstance(layer['pi'], list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                policy_only_layers = layer['pi']

            if 'vf' in layer:
                assert isinstance(layer['vf'], list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                value_only_layers = layer['vf']
            break  # From here on the network splits up in policy and value network

    # Build the non-shared part of the network
    latent_policy = latent
    latent_value = latent
    for idx, (pi_layer_size, vf_layer_size) in enumerate(zip_longest(policy_only_layers, value_only_layers)):
        if pi_layer_size is not None:
            assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
            latent_policy = act_fun(linear(latent_policy, "pi_fc{}".format(idx), pi_layer_size, init_scale=np.sqrt(2)))

        if vf_layer_size is not None:
            assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
            latent_value = act_fun(linear(latent_value, "vf_fc{}".format(idx), vf_layer_size, init_scale=np.sqrt(2)))

    return latent_policy, latent_value



class BasePolicy(ABC):
    """
    The base policy object

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batches to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param scale: (bool) whether or not to scale the input
    :param obs_phs: (TensorFlow Tensor, TensorFlow Tensor) a tuple containing an override for observation placeholder
        and the processed observation placeholder respectively
    :param add_action_ph: (bool) whether or not to create an action placeholder
    """

    recurrent = False

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, scale=False,
                 obs_phs=None, add_action_ph=False):
        self.n_env = n_env
        self.n_steps = n_steps
        self.n_batch = n_batch
        with tf.compat.v1.variable_scope("input", reuse=False):
            if obs_phs is None:
                self._obs_ph, self._processed_obs = observation_input(ob_space, n_batch, scale=scale)
            else:
                self._obs_ph, self._processed_obs = obs_phs

            self._action_ph = None
            if add_action_ph:
                self._action_ph = tf.compat.v1.placeholder(dtype=ac_space.dtype, shape=(n_batch,) + ac_space.shape,
                                                 name="A")
                #self._action_ph = tf.compat.v1.placeholder(dtype=ac_space.dtype, shape=(n_batch,) + ac_space.shape,
                                                           #name="action_ph")
                #X = tf.compat.v1.placeholder(tf.float32, shape=(n_batch,) + ac_space.shape, name='Ob')  # obs
        self.sess = sess
        self.reuse = reuse
        self.ob_space = ob_space
        self.ac_space = ac_space

    @property
    def is_discrete(self):
        """bool: is action space discrete."""
        return isinstance(self.ac_space, Discrete)

    @property
    def initial_state(self):
        """
        The initial state of the policy. For feedforward policies, None. For a recurrent policy,
        a NumPy array of shape (self.n_env, ) + state_shape.
        """
        assert not self.recurrent, "When using recurrent policies, you must overwrite `initial_state()` method"
        return None

    @property
    def obs_ph(self):
        """tf.Tensor: placeholder for observations, shape (self.n_batch, ) + self.ob_space.shape."""
        return self._obs_ph

    @property
    def processed_obs(self):
        """tf.Tensor: processed observations, shape (self.n_batch, ) + self.ob_space.shape.

        The form of processing depends on the type of the observation space, and the parameters
        whether scale is passed to the constructor; see observation_input for more information."""
        return self._processed_obs

    @property
    def action_ph(self):
        """tf.Tensor: placeholder for actions, shape (self.n_batch, ) + self.ac_space.shape."""
        return self._action_ph

    @staticmethod
    def _kwargs_check(feature_extraction, kwargs):
        """
        Ensure that the user is not passing wrong keywords
        when using policy_kwargs.

        :param feature_extraction: (str)
        :param kwargs: (dict)
        """
        # When using policy_kwargs parameter on model creation,
        # all keywords arguments must be consumed by the policy constructor except
        # the ones for the cnn_extractor network (cf nature_cnn()), where the keywords arguments
        # are not passed explicitly (using **kwargs to forward the arguments)
        # that's why there should be not kwargs left when using the mlp_extractor
        # (in that case the keywords arguments are passed explicitly)
        if feature_extraction == 'mlp' and len(kwargs) > 0:
            raise ValueError("Unknown keywords for policy: {}".format(kwargs))

    @abstractmethod
    def step(self, obs, state=None, mask=None):
        """
        Returns the policy for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float], [float], [float], [float]) actions, values, states, neglogp
        """
        raise NotImplementedError

    @abstractmethod
    def proba_step(self, obs, state=None, mask=None):
        """
        Returns the action probability for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float]) the action probability
        """
        raise NotImplementedError


class ActorCriticPolicy(BasePolicy):
    """
    Policy object that implements actor critic

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param scale: (bool) whether or not to scale the input
    """
    
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, scale=False):
        super(ActorCriticPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                                scale=scale)
        self._pdtype = make_proba_dist_type(ac_space)
        self._policy = None
        self._proba_distribution = None
        self._value_fn = None
        self._action = None
        self._deterministic_action = None
        self.multiplier = STDEV_FEATURE
        self.addition = AVG_FEATURE
        
  
    def _setup_init(self):
        """Sets up the distributions, actions, and value."""
        with tf.compat.v1.variable_scope("output", reuse=True):
            assert self.policy is not None and self.proba_distribution is not None and self.value_fn is not None
            self._action = self.proba_distribution.sample()
            self._deterministic_action = self.proba_distribution.mode()
            self._neglogp = self.proba_distribution.neglogp(self.action)
            if isinstance(self.proba_distribution, CategoricalProbabilityDistribution):
                self._policy_proba = tf.nn.softmax(self.policy)
            elif isinstance(self.proba_distribution, DiagGaussianProbabilityDistribution):
                self._policy_proba = [self.proba_distribution.mean, self.proba_distribution.std]
            elif isinstance(self.proba_distribution, BernoulliProbabilityDistribution):
                self._policy_proba = tf.nn.sigmoid(self.policy)
            elif isinstance(self.proba_distribution, MultiCategoricalProbabilityDistribution):
                self._policy_proba = [tf.nn.softmax(categorical.flatparam())
                                     for categorical in self.proba_distribution.categoricals]
            else:
                self._policy_proba = []  # it will return nothing, as it is not implemented
            self._value_flat = self.value_fn[:, 0]

    def _update_preprocess(self, avg_feat, std_feat): 
         self.addition = avg_feat
         self.multiplier = std_feat

    @property
    def pdtype(self):
        """ProbabilityDistributionType: type of the distribution for stochastic actions."""
        return self._pdtype

    @property
    def policy(self):
        """tf.Tensor: policy output, e.g. logits."""
        return self._policy

    @property
    def proba_distribution(self):
        """ProbabilityDistribution: distribution of stochastic actions."""
        return self._proba_distribution

    @property
    def value_fn(self):
        """tf.Tensor: value estimate, of shape (self.n_batch, 1)"""
        return self._value_fn

    @property
    def value_flat(self):
        """tf.Tensor: value estimate, of shape (self.n_batch, )"""
        return self._value_flat

    @property
    def action(self):
        """tf.Tensor: stochastic action, of shape (self.n_batch, ) + self.ac_space.shape."""
        return self._action

    @property
    def deterministic_action(self):
        """tf.Tensor: deterministic action, of shape (self.n_batch, ) + self.ac_space.shape."""
        return self._deterministic_action

    @property
    def neglogp(self):
        """tf.Tensor: negative log likelihood of the action sampled by self.action."""
        return self._neglogp

    @property
    def policy_proba(self):
        """tf.Tensor: parameters of the probability distribution. Depends on pdtype."""
        return self._policy_proba

    @abstractmethod
    def step(self, obs, state=None, mask=None, deterministic=False):
        """
        Returns the policy for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :param deterministic: (bool) Whether or not to return deterministic actions.
        :return: ([float], [float], [float], [float]) actions, values, states, neglogp
        """
        raise NotImplementedError

    @abstractmethod
    def value(self, obs, state=None, mask=None):
        """
        Returns the value for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float]) The associated value of the action
        """
        raise NotImplementedError



class RecurrentActorCriticPolicy(ActorCriticPolicy):
    """
    Actor critic policy object uses a previous state in the computation for the current step.
    NOTE: this class is not limited to recurrent neural network policies,
    see https://github.com/hill-a/stable-baselines/issues/241

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param state_shape: (tuple<int>) shape of the per-environment state space.
    :param reuse: (bool) If the policy is reusable or not
    :param scale: (bool) whether or not to scale the input
    """

    recurrent = True

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch,
                 state_shape, reuse=False, scale=False):
        super(RecurrentActorCriticPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps,
                                                         n_batch, reuse=reuse, scale=scale)

        with tf.variable_scope("input", reuse=False):
            self._dones_ph = tf.compat.v1.placeholder(tf.float32, (n_batch, ), name="dones_ph")  # (done t-1)
            state_ph_shape = (self.n_env, ) + tuple(state_shape)
            self._states_ph = tf.compat.v1.placeholder(tf.float32, state_ph_shape, name="states_ph")

        initial_state_shape = (self.n_env, ) + tuple(state_shape)
        self._initial_state = np.zeros(initial_state_shape, dtype=np.float32)

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def dones_ph(self):
        """tf.Tensor: placeholder for whether episode has terminated (done), shape (self.n_batch, ).
        Internally used to reset the state before the next episode starts."""
        return self._dones_ph

    @property
    def states_ph(self):
        """tf.Tensor: placeholder for states, shape (self.n_env, ) + state_shape."""
        return self._states_ph

    @abstractmethod
    def value(self, obs, state=None, mask=None):
        """
        Cf base class doc.
        """
        raise NotImplementedError


class LstmPolicy(RecurrentActorCriticPolicy):
    """
    Policy object that implements actor critic, using LSTMs.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_lstm: (int) The number of LSTM cells (for recurrent policies)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) The size of the Neural network before the LSTM layer  (if None, default to [64, 64])
    :param net_arch: (list) Specification of the actor-critic policy network architecture. Notation similar to the
        format described in mlp_extractor but with additional support for a 'lstm' entry in the shared network part.
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param layer_norm: (bool) Whether or not to use layer normalizing LSTMs
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    recurrent = True

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False, layers=None,
                 net_arch=None, act_fun=tf.tanh, cnn_extractor=nature_cnn, layer_norm=False, feature_extraction="cnn",
                 **kwargs):
        # state_shape = [n_lstm * 2] dim because of the cell and hidden states of the LSTM
        super(LstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch,
                                         state_shape=(2 * n_lstm, ), reuse=reuse,
                                         scale=(feature_extraction == "cnn"))

        self._kwargs_check(feature_extraction, kwargs)

        if net_arch is None:  # Legacy mode
            if layers is None:
                layers = [64, 64]
            else:
                warnings.warn("The layers parameter is deprecated. Use the net_arch parameter instead.")

            with tf.variable_scope("model", reuse=reuse):
                if feature_extraction == "cnn":
                    extracted_features = cnn_extractor(self.processed_obs, **kwargs)
                else:
                    extracted_features = tf.layers.flatten(self.processed_obs)
                    for i, layer_size in enumerate(layers):
                        extracted_features = act_fun(linear(extracted_features, 'pi_fc' + str(i), n_hidden=layer_size,
                                                            init_scale=np.sqrt(2)))
                input_sequence = batch_to_seq(extracted_features, self.n_env, n_steps)
                masks = batch_to_seq(self.dones_ph, self.n_env, n_steps)
                rnn_output, self.snew = lstm(input_sequence, masks, self.states_ph, 'lstm1', n_hidden=n_lstm,
                                             layer_norm=layer_norm)
                rnn_output = seq_to_batch(rnn_output)
                value_fn = linear(rnn_output, 'vf', 1)

                self._proba_distribution, self._policy, self.q_value = \
                    self.pdtype.proba_distribution_from_latent(rnn_output, rnn_output)

            self._value_fn = value_fn
        else:  # Use the new net_arch parameter
            if layers is not None:
                warnings.warn("The new net_arch parameter overrides the deprecated layers parameter.")
            if feature_extraction == "cnn":
                raise NotImplementedError()

            with tf.variable_scope("model", reuse=reuse):
                latent = tf.layers.flatten(self.processed_obs)
                policy_only_layers = []  # Layer sizes of the network that only belongs to the policy network
                value_only_layers = []  # Layer sizes of the network that only belongs to the value network

                # Iterate through the shared layers and build the shared parts of the network
                lstm_layer_constructed = False
                for idx, layer in enumerate(net_arch):
                    if isinstance(layer, int):  # Check that this is a shared layer
                        layer_size = layer
                        latent = act_fun(linear(latent, "shared_fc{}".format(idx), layer_size, init_scale=np.sqrt(2)))
                    elif layer == "lstm":
                        if lstm_layer_constructed:
                            raise ValueError("The net_arch parameter must only contain one occurrence of 'lstm'!")
                        input_sequence = batch_to_seq(latent, self.n_env, n_steps)
                        masks = batch_to_seq(self.dones_ph, self.n_env, n_steps)
                        rnn_output, self.snew = lstm(input_sequence, masks, self.states_ph, 'lstm1', n_hidden=n_lstm,
                                                     layer_norm=layer_norm)
                        latent = seq_to_batch(rnn_output)
                        lstm_layer_constructed = True
                    else:
                        assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
                        if 'pi' in layer:
                            assert isinstance(layer['pi'],
                                              list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                            policy_only_layers = layer['pi']

                        if 'vf' in layer:
                            assert isinstance(layer['vf'],
                                              list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                            value_only_layers = layer['vf']
                        break  # From here on the network splits up in policy and value network

                # Build the non-shared part of the policy-network
                latent_policy = latent
                for idx, pi_layer_size in enumerate(policy_only_layers):
                    if pi_layer_size == "lstm":
                        raise NotImplementedError("LSTMs are only supported in the shared part of the policy network.")
                    assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
                    latent_policy = act_fun(
                        linear(latent_policy, "pi_fc{}".format(idx), pi_layer_size, init_scale=np.sqrt(2)))

                # Build the non-shared part of the value-network
                latent_value = latent
                for idx, vf_layer_size in enumerate(value_only_layers):
                    if vf_layer_size == "lstm":
                        raise NotImplementedError("LSTMs are only supported in the shared part of the value function "
                                                  "network.")
                    assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
                    latent_value = act_fun(
                        linear(latent_value, "vf_fc{}".format(idx), vf_layer_size, init_scale=np.sqrt(2)))

                if not lstm_layer_constructed:
                    raise ValueError("The net_arch parameter must contain at least one occurrence of 'lstm'!")

                self._value_fn = linear(latent_value, 'vf', 1)
                # TODO: why not init_scale = 0.001 here like in the feedforward
                self._proba_distribution, self._policy, self.q_value = \
                    self.pdtype.proba_distribution_from_latent(latent_policy, latent_value)
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            return self.sess.run([self.deterministic_action, self.value_flat, self.snew, self.neglogp],
                                 {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})
        else:
            return self.sess.run([self.action, self.value_flat, self.snew, self.neglogp],
                                 {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})


class FeedForwardPolicy(ActorCriticPolicy):
    """
    Policy object that implements actor critic, using a feed forward neural network.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) (deprecated, use net_arch instead) The size of the Neural network for the policy
        (if None, default to [64, 64])
    :param net_arch: (list) Specification of the actor-critic policy network architecture (see mlp_extractor
        documentation for details).
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None, net_arch=None,
                 act_fun=tf.tanh, cnn_extractor=nature_cnn, feature_extraction="cnn", **kwargs):
        super(FeedForwardPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                                scale=(feature_extraction == "cnn"))


        self._kwargs_check(feature_extraction, kwargs)

        if layers is not None:
            warnings.warn("Usage of the `layers` parameter is deprecated! Use net_arch instead "
                          "(it has a different semantics though).", DeprecationWarning)
            if net_arch is not None:
                warnings.warn("The new `net_arch` parameter overrides the deprecated `layers` parameter!",
                              DeprecationWarning)

        if net_arch is None:
            if layers is None:
                layers = [64, 64]
            net_arch = [dict(vf=layers, pi=layers)]

        with tf.compat.v1.variable_scope("model", reuse=reuse):
            if feature_extraction == "cnn":
                pi_latent = vf_latent = cnn_extractor(self.processed_obs, **kwargs)
            else:
                pi_latent, vf_latent = mlp_extractor(tf.compat.v1.layers.flatten(self.processed_obs), net_arch, act_fun)

            self._value_fn = linear(vf_latent, 'vf', 1)

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})



class CnnPolicy(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a CNN (the nature CNN)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(CnnPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        feature_extraction="cnn", **_kwargs)


class CnnLstmPolicy(LstmPolicy):
    """
    Policy object that implements actor critic, using LSTMs with a CNN feature extraction

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_lstm: (int) The number of LSTM cells (for recurrent policies)
    :param reuse: (bool) If the policy is reusable or not
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False, **_kwargs):
        super(CnnLstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                                            layer_norm=False, feature_extraction="cnn", **_kwargs)


class CnnLnLstmPolicy(LstmPolicy):
    """
    Policy object that implements actor critic, using a layer normalized LSTMs with a CNN feature extraction

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_lstm: (int) The number of LSTM cells (for recurrent policies)
    :param reuse: (bool) If the policy is reusable or not
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False, **_kwargs):
        super(CnnLnLstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                                              layer_norm=True, feature_extraction="cnn", **_kwargs)


class MlpPolicy(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a MLP (2 layers of 64)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(MlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        feature_extraction="mlp", **_kwargs)
        



class MlpLstmPolicy(LstmPolicy):
    """
    Policy object that implements actor critic, using LSTMs with a MLP feature extraction

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_lstm: (int) The number of LSTM cells (for recurrent policies)
    :param reuse: (bool) If the policy is reusable or not
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False, **_kwargs):
        super(MlpLstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                                            layer_norm=False, feature_extraction="mlp", **_kwargs)


class MlpLnLstmPolicy(LstmPolicy):
    """
    Policy object that implements actor critic, using a layer normalized LSTMs with a MLP feature extraction

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_lstm: (int) The number of LSTM cells (for recurrent policies)
    :param reuse: (bool) If the policy is reusable or not
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False, **_kwargs):
        super(MlpLnLstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                                              layer_norm=True, feature_extraction="mlp", **_kwargs)


        

########################################################################################################################
# Intrinsic Reward Augmented Policies + Innovation r(s,a,s')
########################################################################################################################
class MlpPolicyIntrinsicInnovationReward(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, avg_feature, std_feature, reuse=False): #pylint: disable=W0613
        ob_shape = (nbatch,) + ob_space.shape #(5,181)
        actdim = ac_space.shape[0] #30 
        X = tf.compat.v1.placeholder(tf.float32, ob_shape, name='Ob')  # obs
        addition = tf.compat.v1.constant(avg_feature, dtype=tf.float32, name='addition')  #tensor shape (n_step, 181)
        multiplier = tf.compat.v1.constant(std_feature, dtype=tf.float32, name='multiplier')
        X_preproc = (X - addition) * multiplier 
        #print("shape of X:", X)
        

        with tf.compat.v1.variable_scope('policy', reuse=reuse):
            activ = tf.tanh
            
            h1 = activ(fc(X_preproc, 'v_mix_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, 'v_mix_fc2', nh=64, init_scale=np.sqrt(2)))
            v_mix0 = fc(h2, 'v_mix', 1)[:,0]
            logstd = tf.compat.v1.get_variable(name="logstd", shape=[1, actdim], initializer=tf.zeros_initializer()) # trainable. 
            h1_ = activ(fc(X_preproc, 'pi_fc1', nh=64, init_scale=np.sqrt(2)))
            h2_ = activ(fc(h1_, 'pi_fc2', nh=64, init_scale=np.sqrt(2)))
            pi = fc(h2_, 'pi', actdim, init_scale=0.01) # length 30.  Determines action for each of the 30 stocks.
            
        with tf.compat.v1.variable_scope('intrinsic', reuse=reuse):
            X_NX = tf.compat.v1.placeholder(tf.float32, (None,) + ob_space.shape, name='Ob_all') #obs  (None ,181)
            X_NX_preproc = (X_NX - addition) * multiplier 
            A_ALL = tf.compat.v1.placeholder(tf.float32, [None, actdim], name='Ac_all')
            A_ALL_multiplier = tf.compat.v1.constant(10, dtype=tf.float32, shape=[nbatch,actdim]) 
            A_ALL_preproc = A_ALL * A_ALL_multiplier 
            
            INPUT = tf.concat([X_preproc, A_ALL_preproc, X_NX_preproc], axis=1)
            activ = tf.tanh
            h1 = activ(fc(X_preproc, 'v_ex_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, 'v_ex_fc2', nh=64, init_scale=np.sqrt(2)))
            v_ex0 = fc(h2, 'v_ex', 1)[:,0]
            
            h1_ = activ(fc(INPUT, 'intrinsic_fc1', nh=64, init_scale=np.sqrt(2)))
            #h1 = activ(fc(X, 'intrinsic_fc1', nh=64, init_scale=np.sqrt(2)))
            h2_ = activ(fc(h1_, 'intrinsic_fc2', nh=64, init_scale=np.sqrt(2)))
            r_in0 = tf.tanh(fc(h2_, 'r_in', 1))[:,0]    #r_in(s,a): estimated by eta, here the params eta refers to this layer of NN
            

        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1) #length 60

        self.pdtype = make_proba_dist_type(ac_space) #<stable_baselines.common.distributions.DiagGaussianProbabilityDistributionType object >

        self.pd = self.pdtype.proba_distribution_from_flat(pdparam) # make_stochastic_policy(pdparam)

        a0 = self.pd.sample() # sample from pd of pi
        #print("a0:", a0)
        neglogp0 = self.pd.neglogp(a0) 
        self.initial_state = None
        
        def test(ob): 
            logstd_, pi_, pdparam_, a0_, neglogp0_ =sess.run([logstd, pi, pdparam, a0, neglogp0],{X:ob})
            return logstd_, pi_, pdparam_, a0_, neglogp0_

        def step(ob, *_args, **_kwargs):
            a, v_ex, v_mix, neglogp = sess.run([a0, v_ex0, v_mix0, neglogp0], {X:ob})
            return a, v_ex, v_mix, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            v_ex, v_mix = sess.run([v_ex0, v_mix0], {X:ob})
            return v_ex, v_mix

        def intrinsic_reward(ob, ac, ob_nx = None, *_args, **_kwargs):    ###INPUT INCLUDE OB_NX
            #if ac is not None and ob_nx is not None:
            #r_in = .. X:ob, A_ALL: ac, X_NX: ob_nx
            #elif ob_nx is None:
            # r_in = .. X:ob, A_ALL: ac
            #else:
            # r_in = .. X:ob
            if ob_nx is None:
                r_in = sess.run(r_in0, {X: ob, A_ALL: ac})
            else:
                r_in = sess.run(r_in0, {X:ob, A_ALL:ac, X_NX:ob_nx})    ###CODE TO EXPEND AND INCLUDE X_NX
            #print("r_in:", r_in)
            return r_in

        def proba_step(ob, state=None, mask=None):
            print("PROBA STEP RAN.")
            return sess.run(self.policy_proba, {self.obs_ph: ob})

        self.X = X
        self.X_preproc = X_preproc
        self.X_NX = X_NX
        self.A_ALL = A_ALL
        self.pi = pi
        self.v_ex = v_ex0
        self.r_in = r_in0
        self.v_mix = v_mix0
        self.step = step
        self.value = value
        self.test= test 
        self.intrinsic_reward = intrinsic_reward
        self.policy_params = tf.compat.v1.trainable_variables("policy")
        self.intrinsic_params = tf.compat.v1.trainable_variables("intrinsic")
        self.policy_new_fn = MlpPolicyNewInnovation
    


class MlpPolicyIntrinsicInnovationReward_PPO(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, avg_feature, std_feature, reuse=False): #pylint: disable=W0613
        ob_shape = (nbatch,) + ob_space.shape 
        "step model: (1, 181)     train_model: (5,181) "
        actdim = ac_space.shape[0] #30 
        addition = tf.compat.v1.constant(avg_feature, dtype=tf.float32, name='addition')  # (nbatch, 181) = (1, 181) or (5,181) 
        multiplier = tf.compat.v1.constant(std_feature, dtype=tf.float32, name='multiplier')
        X = tf.compat.v1.placeholder(tf.float32, ob_shape, name='Ob')  # obs
        X_preproc = (X - addition) * multiplier 
        
        
        #print("shape of X:", X)
        

        with tf.compat.v1.variable_scope('policy', reuse=reuse):
            
            activ = tf.tanh
            
            logstd = tf.compat.v1.get_variable(name="logstd", shape=[1, actdim], initializer=tf.zeros_initializer()) # trainable. 
            
            h1 = activ(fc(X_preproc, 'v_mix_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, 'v_mix_fc2', nh=64, init_scale=np.sqrt(2)))
            v_mix0 = fc(h2, 'v_mix', 1)[:,0]
            
            h1 = activ(fc(X_preproc, 'pi_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2)))
            pi = fc(h2, 'pi', actdim, init_scale=0.01) # length 30.  Determines action for each of the 30 stocks.
            
        with tf.compat.v1.variable_scope('intrinsic', reuse=reuse):
            X_ALL = tf.compat.v1.placeholder(tf.float32, (None,) + ob_space.shape, name='Ob_all') # (nsteos, 181) = (1 , 181 )  or (20,181 )
            addition_all = tf.tile(addition, [nsteps//nbatch , 1])
            multiplier_all = tf.tile(multiplier, [nsteps//nbatch , 1])
            X_ALL_preproc = (X_ALL - addition_all )*multiplier_all 
            
            A_ALL = tf.compat.v1.placeholder(tf.float32, [None, actdim], name='Ac_all') # 
            A_ALL_multiplier = tf.compat.v1.constant(5, dtype=tf.float32, shape=[nsteps,actdim])  # (nsteos, 181) = (1 , 181 )  or (20,181 ) 
            A_ALL_preproc = A_ALL * A_ALL_multiplier 
            
            X_NX = tf.compat.v1.placeholder(tf.float32, (None,) + ob_space.shape, name='Ob_nx') # (nsteos, 181) = (1 , 181 )  or (20,181 ) 
            X_NX_preproc = (X_NX - addition_all) * multiplier_all 
            
            
            INPUT = tf.concat([X_ALL_preproc, A_ALL_preproc, X_NX_preproc], axis=1)
            activ = tf.tanh
            h1 = activ(fc(X_preproc, 'v_ex_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, 'v_ex_fc2', nh=64, init_scale=np.sqrt(2)))
            v_ex0 = fc(h2, 'v_ex', 1)[:,0]
            
            h1_ = activ(fc(INPUT, 'intrinsic_fc1', nh=64, init_scale=np.sqrt(2)))
            #h1 = activ(fc(X, 'intrinsic_fc1', nh=64, init_scale=np.sqrt(2)))
            h2_ = activ(fc(h1_, 'intrinsic_fc2', nh=64, init_scale=np.sqrt(2)))
            r_in0 = tf.tanh(fc(h2_, 'r_in', 1))[:,0]    #r_in(s,a): estimated by eta, here the params eta refers to this layer of NN
            

        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1) #length 60

        self.pdtype = make_proba_dist_type(ac_space) #<stable_baselines.common.distributions.DiagGaussianProbabilityDistributionType object >

        self.pd = self.pdtype.proba_distribution_from_flat(pdparam) # make_stochastic_policy(pdparam)

        a0 = self.pd.sample() # sample from pd of pi
        #print("a0:", a0)
        neglogp0 = self.pd.neglogp(a0) 
        self.initial_state = None
        
        def test(ob): 
            logstd_, pi_, pdparam_, a0_, neglogp0_ =sess.run([logstd, pi, pdparam, a0, neglogp0],{X:ob})
            return logstd_, pi_, pdparam_, a0_, neglogp0_

        def step(ob, *_args, **_kwargs):
            a, v_ex, v_mix, neglogp = sess.run([a0, v_ex0, v_mix0, neglogp0], {X:ob})
            return a, v_ex, v_mix, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            v_ex, v_mix = sess.run([v_ex0, v_mix0], {X:ob})
            return v_ex, v_mix

        def intrinsic_reward(ob, ac, ob_nx = None, *_args, **_kwargs):    
            r_in = sess.run(r_in0, {X_ALL:ob, A_ALL:ac, X_NX:ob_nx})    
            return r_in

        def proba_step(ob, state=None, mask=None):
            print("PROBA STEP RAN.")
            return sess.run(self.policy_proba, {self.obs_ph: ob})

        self.X = X
        self.X_preproc = X_preproc
        self.X_ALL = X_ALL 
        self.X_NX = X_NX
        self.A_ALL = A_ALL
        self.pi = pi
        self.v_ex = v_ex0
        self.r_in = r_in0
        self.v_mix = v_mix0
        self.step = step
        self.value = value
        self.test= test 
        self.intrinsic_reward = intrinsic_reward
        self.policy_params = tf.compat.v1.trainable_variables("policy")
        self.intrinsic_params = tf.compat.v1.trainable_variables("intrinsic")
        self.policy_new_fn = MlpPolicyNewInnovation
    






class MlpPolicyNewInnovation(object):
    def __init__(self, params, ob_space, ac_space, nbatch, nsteps, avg_feature, std_feature): #pylint: disable=W0613
        ob_shape = (nbatch,) + ob_space.shape       
        addition = tf.compat.v1.constant(avg_feature, dtype=tf.float32, name='addition')  #tensor shape (n_step, 181)
        multiplier = tf.compat.v1.constant(std_feature, dtype=tf.float32, name='multiplier')

        
        X = tf.compat.v1.placeholder(tf.float32, ob_shape, name='Ob') #obs
        X_preproc = (X - addition) * multiplier 
        with tf.name_scope('policy_new'):
            activ = tf.tanh 
            h1 = activ(tf.compat.v1.nn.xw_plus_b(X_preproc, params['policy/pi_fc1/w:0'], params['policy/pi_fc1/b:0']))
            h2 = activ(tf.compat.v1.nn.xw_plus_b(h1, params['policy/pi_fc2/w:0'], params['policy/pi_fc2/b:0']))
            pi = tf.compat.v1.nn.xw_plus_b(h2, params['policy/pi/w:0'], params['policy/pi/b:0'])
            logstd = params['policy/logstd:0']


        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)

        self.pdtype = make_proba_dist_type(ac_space)
        self.pd = self.pdtype.proba_distribution_from_flat(pdparam)

        self.X = X







def batch_to_seq(h, nbatch, nsteps, flat=False):
    if flat:
        h = tf.reshape(h, [nbatch, nsteps])
    else:
        h = tf.reshape(h, [nbatch, nsteps, -1]) 
    # Squeeze each tensor along the time step dimension
    # This removes the time step dimension from each tensor
    # resulting in a list of tensors, each representing a time step
    return [tf.squeeze(v, [1]) for v in tf.split(axis=1, num_or_size_splits=nsteps, value=h)]

def seq_to_batch(h, flat = False):
    shape = h[0].get_shape().as_list()
    if not flat:
        assert(len(shape) > 1)
        nh = h[0].get_shape()[-1]        
        return tf.reshape(tf.concat(axis=1, values=h), [-1, nh])
    else:
        return tf.reshape(tf.stack(values=h, axis=1), [-1])


def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init





########################################################################################################################
# Intrinsic Reward Policy for Autoregressive with LSTM 
########################################################################################################################

class MlpPolicyIntrinsicInnovationReward_auto_R(object): #MlpPolicyIntrinsicInnovationReward_v6_1
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, avg_feature, std_feature, nlstm=30,reuse=False): 
        nenv = nbatch // nsteps
        ob_shape = (nsteps, 182) 
        actdim = 1
        self.pdtype = make_proba_dist_type(ac_space)
        
                
        M = tf.compat.v1.placeholder(tf.float32, [nbatch], name='Mask')
        X = tf.compat.v1.placeholder(tf.float32, ob_shape, name='Ob')  # INPUT: (State, Prev_action) 
        
        
        
        addition = tf.compat.v1.constant(avg_feature, dtype=tf.float32, name='addition')  
        multiplier = tf.compat.v1.constant(std_feature, dtype=tf.float32, name='multiplier')
        X_preproc = (X - addition) * multiplier 
        xs = batch_to_seq(X_preproc, nenv, nsteps)         
        ms = batch_to_seq(M, nenv, nsteps )
        

        def lstm(xs, ms, s, scope, nh, init_scale=np.sqrt(2)): # xs: list of len (nsteps*30). each item (1,182) # ms: list of len (nsteps*30). each item (1,1). s: hidden state. shape ()
            nbatch, nin = [v for v in xs[0].get_shape()] # (1, 182 )     
            with tf.compat.v1.variable_scope(scope):
                wx = tf.compat.v1.get_variable("wx", [nin, nh*4], initializer=ortho_init(init_scale)) 
                wh = tf.compat.v1.get_variable("wh", [nh, nh*4], initializer=ortho_init(init_scale)) 
                b = tf.compat.v1.get_variable("b", [nh*4], initializer=tf.constant_initializer(0.0))
            c, h = tf.split(axis=1, num_or_size_splits=2, value=s) 
            for idx, (x, m) in enumerate(zip(xs, ms)): #idx: each timestep    # x: (1,182)   # m: (1,1)
                c = c*(1-m) 
                h = h*(1-m) 
                z = tf.matmul(x, wx) + tf.matmul(h, wh) + b 
                i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
                i = tf.nn.sigmoid(i) 
                f = tf.nn.sigmoid(f) 
                o = tf.nn.sigmoid(o) 
                u = tf.tanh(u) 
                c = f*c + i*u 
                h = o*tf.tanh(c) 
                xs[idx] = h
            s = tf.concat(axis=1, values=[c, h])
            return xs, s
         
        
        with tf.compat.v1.variable_scope('policy', reuse=reuse):
            S = tf.compat.v1.placeholder(tf.float32, [nenv, nlstm*2], name='State') # hidden state 
            logstd = tf.compat.v1.get_variable(name="logstd", shape=[1, 1], initializer=tf.zeros_initializer())
            activ = tf.tanh
            rnn_output, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)  #rnn_output: list of len nsteps. each item (1,2*nh). snew: final state. 
            rnn_output = seq_to_batch(rnn_output) 
            h1 = activ(fc(rnn_output, 'pi_fc1', nh=64, init_scale=np.sqrt(2)))
            pi = fc(h1, 'pi', actdim, init_scale=0.01) 
            
            S_vf = tf.compat.v1.placeholder(tf.float32, [nenv, nlstm*2], name='State_vf') # hidden state 
            rnn_output_, snew_vf = lstm(xs, ms, S_vf, 'lstm2', nh=nlstm)  #rnn_output: list of len nsteps. each item (1,2*nh). snew: final state. 
            rnn_output_ = seq_to_batch(rnn_output_) # (nsteps, 2*nh)
            h1 = activ(fc(rnn_output_, 'v_mix_fc1', nh=64, init_scale=np.sqrt(2)))
            v_mix0 = fc(h1, 'v_mix', 1)[:,0]
            h1 = activ(fc(rnn_output_, 'v_ex_fc1', nh=64, init_scale=np.sqrt(2)))
            v_ex0 = fc(h1, 'v_ex', 1)[:,0]
           
                
        with tf.compat.v1.variable_scope('intrinsic', reuse=reuse):
            activ = tf.tanh
            X_NX = tf.compat.v1.placeholder(tf.float32, ob_shape , name='Ob_all') # (none , 211) Next_state
            X_NX_preproc =( X_NX - addition ) *  multiplier 
            A_ALL = tf.compat.v1.placeholder(tf.float32, [None, actdim], name='Ac_all')  # Recent action. 
            INPUT = tf.concat([X_preproc, 5*A_ALL, X_NX_preproc], axis=1)
        
            S_r_in = tf.compat.v1.placeholder(tf.float32, [nenv, nlstm*2], name='State_rin') # hidden state 

            
            xs_r_in = batch_to_seq(INPUT, nenv, nsteps) # list of len nsteps . each item (1,183)
            ms_r_in = batch_to_seq(M, nenv, nsteps) 
            rnn_output, snew_rin = lstm(xs_r_in, ms_r_in, S_r_in, 'lstm_r_in', nh=nlstm)  
            rnn_output = seq_to_batch(rnn_output) # (nsteps, 2nh)
            
            h1 = activ(fc(rnn_output, 'intrinsic_fc1', nh=64, init_scale=np.sqrt(2)))
            r_in0 = tf.tanh(fc(h1, 'r_in', 1))[:,0] 
            
            
   
            
        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)
        self.pd = self.pdtype.proba_distribution_from_flat(pdparam)
        a0 = self.pd.sample()
        a0 = tf.compat.v1.clip_by_value(a0, clip_value_min=-1, clip_value_max=1)
        neglogp0 = self.pd.neglogp(a0)
    
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)
        
        

        def step(ob, state ,s_vf, mask ): # Ob : (S_t , a_{t-1} ) 
            a, v_ex, v_mix, snew_, s_vf_new, neglogp = sess.run([a0, v_ex0, v_mix0, snew,snew_vf, neglogp0], {X:ob, S:state,S_vf:s_vf, M:mask})
            
            
            return a, v_ex, v_mix, snew_, s_vf_new, neglogp

        def value(ob, s, s_vf, mask):
            v_ex, v_mix,s_vf_new = sess.run([v_ex0, v_mix0, snew_vf ], {X:ob, S:s ,S_vf:s_vf, M:mask}) 
            
            
            return v_ex, v_mix,s_vf_new

        def intrinsic_reward(ob, ac, state, mask , ob_nx = None, *_args, **_kwargs):    ###INPUT INCLUDE OB_NX 
            r_in ,snew_rin_ = sess.run([r_in0 , snew_rin ], {X:ob, A_ALL:ac, X_NX:ob_nx,S_r_in:state , M:mask })    ###CODE TO EXPEND AND INCLUDE X_NX
            return r_in, snew_rin_ 
        
        
        

        def proba_step(ob, state=None, mask=None):
            return sess.run(self.policy_proba, {self.obs_ph: ob})

        self.X = X
        self.A_ALL = A_ALL
        self.pi = pi
        self.v_ex = v_ex0
        self.r_in = r_in0
        self.M = M 
        self.S = S 
        self.S_vf =S_vf 
        self.X_NX = X_NX
        self.S_r_in = S_r_in 
        self.v_mix = v_mix0
        self.step = step
        self.value = value
        self.intrinsic_reward = intrinsic_reward
        self.policy_params = tf.compat.v1.trainable_variables("policy")
        self.intrinsic_params = tf.compat.v1.trainable_variables("intrinsic")
        self.policy_new_fn = MlpPolicyNewInnovation_auto_R #MlpPolicyNewInnovation_v6_1
        




class MlpPolicyNewInnovation_auto_R(object):
    def __init__(self, params, ob_space, ac_space, nbatch, nsteps, avg_feature, std_feature,nlstm=30): #pylint: disable=W0613
        "avg_feature, std_feature: (n_steps, 181) array "
        nenv = nbatch // nsteps
        ob_shape = (nsteps, 182) # (1,182) or (30,182) 
        actdim = 1
        self.pdtype = make_proba_dist_type(ac_space)
        
                
        M = tf.compat.v1.placeholder(tf.float32, [nbatch], name='Mask')
        X = tf.compat.v1.placeholder(tf.float32, ob_shape, name='Ob')  # INPUT: (State, Prev_action) 
        S = tf.compat.v1.placeholder(tf.float32, [nenv, nlstm*2], name='State') # hidden state 
        
        
        
        addition = tf.compat.v1.constant(avg_feature, dtype=tf.float32, name='addition')  #tensor shape (nsteps, 182)
        multiplier = tf.compat.v1.constant(std_feature, dtype=tf.float32, name='multiplier')
        X_preproc = (X - addition) * multiplier 
        xs = batch_to_seq(X_preproc, nenv, nsteps) # list of len nsteps . each item (1,182)
        ms = batch_to_seq(M, nenv, nsteps )
        #print("shape of X:", X)
    
    
        def lstm(xs, ms, s, scope, nh, init_scale=np.sqrt(2)): # xs: list of len (nsteps*30). each item (1,182) # ms: list of len (nsteps*30). each item (1,1). s: hidden state. shape ()
            nbatch, nin = [v for v in xs[0].get_shape()] # (1, 182 )     
            with tf.compat.v1.variable_scope(scope):
                wx = params['policy/lstm1/wx:0']
                wh = params['policy/lstm1/wh:0']
                b = params['policy/lstm1/b:0']
            c, h = tf.split(axis=1, num_or_size_splits=2, value=s) 
            for idx, (x, m) in enumerate(zip(xs, ms)): #idx: each timestep    # x: (1,182)   # m: (1,1)
                c = c*(1-m) 
                h = h*(1-m)                 
                z = tf.matmul(x, wx) + tf.matmul(h, wh) + b 
                i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
                i = tf.nn.sigmoid(i) 
                f = tf.nn.sigmoid(f) 
                o = tf.nn.sigmoid(o) 
                u = tf.tanh(u) 
                c = f*c + i*u 
                h = o*tf.tanh(c) 
                xs[idx] = h
            s = tf.concat(axis=1, values=[c, h])
            return xs, s
    
    
        with tf.name_scope('policy_new'):
            activ = tf.tanh
            logstd = params['policy/logstd:0']
            rnn_output, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)  #rnn_output: list of len nsteps. each item (1,2*nh). snew: final state. 
            rnn_output = seq_to_batch(rnn_output) # (nsteps, 2*nh)
            h1 = activ(tf.compat.v1.nn.xw_plus_b(rnn_output, params['policy/pi_fc1/w:0'], params['policy/pi_fc1/b:0']))
            pi = tf.compat.v1.nn.xw_plus_b(h1, params['policy/pi/w:0'], params['policy/pi/b:0'])
            
            
        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)
        self.pd = self.pdtype.proba_distribution_from_flat(pdparam)
                
        
        self.X = X
        self.M=M
        self.S = S




########################################################################################################################
# Intrinsic Reward Policy for Autoregressive with FNN 
########################################################################################################################

class MlpPolicyIntrinsicInnovationReward_auto_F(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, avg_feature, std_feature,reuse=False): 
        nenv = nbatch // nsteps        
        ob_shape = (nsteps, 211) # 181+30 
        actdim = 1
        
    
        X = tf.compat.v1.placeholder(tf.float32, ob_shape, name='Ob')  # INPUT: (State, 30 Actions ) 

        addition = tf.compat.v1.constant(avg_feature, dtype=tf.float32, name='addition')  #tensor shape (nsteps, 182)
        multiplier = tf.compat.v1.constant(std_feature, dtype=tf.float32, name='multiplier')
        
        X_preproc = (X - addition) * multiplier 
        
        
        with tf.compat.v1.variable_scope('policy', reuse=reuse):
            activ = tf.tanh
            logstd = tf.compat.v1.get_variable(name="logstd", shape=[1, actdim], initializer=tf.zeros_initializer())
            h1 = activ(fc(X_preproc, 'v_mix_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, 'v_mix_fc2', nh=64, init_scale=np.sqrt(2)))
            v_mix0 = fc(h2, 'v_mix', 1)[:,0]
            h1 = activ(fc(X_preproc, 'pi_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2)))
            pi = fc(h2, 'pi', actdim, init_scale=0.01)
        
        

        with tf.compat.v1.variable_scope('intrinsic', reuse=reuse):
            X_NX = tf.compat.v1.placeholder(tf.float32, ob_shape , name='Ob_all') # (none , 211) Next_state
            X_NX_preproc =( X_NX - addition ) *  multiplier 
            A_ALL = tf.compat.v1.placeholder(tf.float32, [None, actdim], name='Ac_all')  # Recent action. 
            INPUT = tf.concat([X_preproc, 5*A_ALL, X_NX_preproc], axis=1)
            
            activ = tf.tanh
            h1 = activ(fc(INPUT, 'intrinsic_fc1', nh=64, init_scale=np.sqrt(2)))
            #h1 = activ(fc(X, 'intrinsic_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, 'intrinsic_fc2', nh=64, init_scale=np.sqrt(2)))
            r_in0 = tf.tanh(fc(h2, 'r_in', 1))[:,0]    
            h1 = activ(fc(X_preproc , 'v_ex_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, 'v_ex_fc2', nh=64, init_scale=np.sqrt(2)))
            v_ex0 = fc(h2, 'v_ex', 1)[:,0]
            
        
            
        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)
        self.pdtype = make_proba_dist_type(ac_space)
        self.pd = self.pdtype.proba_distribution_from_flat(pdparam)
        a0 = self.pd.sample()
        a0 = tf.compat.v1.clip_by_value(a0, clip_value_min=-1, clip_value_max=1)
        neglogp0 = self.pd.neglogp(a0)
        
            

        
        self.initial_state = None 
        
        
        def step(ob, *_args, **_kwargs):
            a, v_ex, v_mix, neglogp = sess.run([a0, v_ex0, v_mix0, neglogp0], {X:ob})
            
            return a, v_ex, v_mix, self.initial_state, neglogp

            
            
        def value(ob, *_args, **_kwargs):
            v_ex, v_mix = sess.run([v_ex0, v_mix0], {X:ob})
            return v_ex, v_mix
        
        
        def intrinsic_reward(ob, ac, ob_nx = None, *_args, **_kwargs): 
            if ob_nx is None:
                r_in = sess.run(r_in0, {X: ob, A_ALL: ac})
            else:
                r_in, intrin_input = sess.run([r_in0,INPUT], {X:ob, A_ALL:ac, X_NX:ob_nx})    ###CODE TO EXPEND AND INCLUDE X_NX
            return r_in

      
                
          
        def proba_step(ob, state=None, mask=None):
            return sess.run(self.policy_proba, {self.obs_ph: ob})


        self.X = X
        self.X_NX = X_NX
        self.A_ALL = A_ALL
        self.pi = pi
        self.v_ex = v_ex0
        self.r_in = r_in0
        self.v_mix = v_mix0
        self.step = step
        self.value = value
        self.intrinsic_reward = intrinsic_reward
        self.policy_params = tf.compat.v1.trainable_variables("policy")
        self.intrinsic_params = tf.compat.v1.trainable_variables("intrinsic")
        self.policy_new_fn = MlpPolicyNewInnovation_auto_F


class MlpPolicyNewInnovation_auto_F(object):
    def __init__(self, params, ob_space, ac_space, nbatch, nsteps, avg_feature, std_feature): #pylint: disable=W0613
        ob_shape = (nsteps, 211) 

        X = tf.compat.v1.placeholder(tf.float32, ob_shape, name='Ob')  # INPUT: (State, 30 Actions ) 

        addition = tf.compat.v1.constant(avg_feature, dtype=tf.float32, name='addition')  #tensor shape (nsteps, 182)
        multiplier = tf.compat.v1.constant(std_feature, dtype=tf.float32, name='multiplier')
        X_preproc = (X - addition) * multiplier     
    
        with tf.name_scope('policy_new'):
            activ = tf.tanh
            h1 = activ(tf.compat.v1.nn.xw_plus_b(X_preproc, params['policy/pi_fc1/w:0'], params['policy/pi_fc1/b:0']))
            h2 = activ(tf.compat.v1.nn.xw_plus_b(h1, params['policy/pi_fc2/w:0'], params['policy/pi_fc2/b:0']))
            pi = tf.compat.v1.nn.xw_plus_b(h2, params['policy/pi/w:0'], params['policy/pi/b:0'])
            logstd = params['policy/logstd:0']

        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)

        self.pdtype = make_proba_dist_type(ac_space)
        self.pd = self.pdtype.proba_distribution_from_flat(pdparam)

        self.X = X







_policy_registry = {
    ActorCriticPolicy: {
        "CnnPolicy": CnnPolicy,
        "CnnLstmPolicy": CnnLstmPolicy,
        "CnnLnLstmPolicy": CnnLnLstmPolicy,
        "MlpPolicy": MlpPolicy,
        "MlpLstmPolicy": MlpLstmPolicy,
        "MlpLnLstmPolicy": MlpLnLstmPolicy,
        "MlpPolicyIntrinsicInnovationReward": MlpPolicyIntrinsicInnovationReward,  
        "MlpPolicyIntrinsicInnovationReward_auto_F": MlpPolicyIntrinsicInnovationReward_auto_F, 
        "MlpPolicyIntrinsicInnovationReward_auto_R": MlpPolicyIntrinsicInnovationReward_auto_R
    }
}


def get_policy_from_name(base_policy_type, name):
    """
    returns the registed policy from the base type and name

    :param base_policy_type: (BasePolicy) the base policy object
    :param name: (str) the policy name
    :return: (base_policy_type) the policy
    """
    if base_policy_type not in _policy_registry:
        raise ValueError("Error: the policy type {} is not registered!".format(base_policy_type))
    if name not in _policy_registry[base_policy_type]:
        raise ValueError("Error: unknown policy type {}, the only registed policy type are: {}!"
                         .format(name, list(_policy_registry[base_policy_type].keys())))
    return _policy_registry[base_policy_type][name]


def register_policy(name, policy):
    """
    returns the registed policy from the base type and name
    :param name: (str) the policy name
    :param policy: (subclass of BasePolicy) the policy
    """
    sub_class = None
    for cls in BasePolicy.__subclasses__():
        if issubclass(policy, cls):
            sub_class = cls
            break
    if sub_class is None:
        raise ValueError("Error: the policy {} is not of any known subclasses of BasePolicy!".format(policy))

    if sub_class not in _policy_registry:
        _policy_registry[sub_class] = {}
    if name in _policy_registry[sub_class]:
        raise ValueError("Error: the name {} is alreay registered for a different policy, will not override."
                         .format(name))
    _policy_registry[sub_class][name] = policy


