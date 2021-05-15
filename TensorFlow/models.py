from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.models import load_model

import os


SMALL_CONST = 1e-5


class ContinuousActorCriticModel(Model):
    """Implementation of an actor-critic model with a continuous action space."""

    def __init__(self, num_observations, num_actions, actor_hidden_units=[256, 256, 256], critic_hidden_units=[256, 256, 256],
                 actor_activation_function='relu', critic_activation_function='relu'):
        super(ContinuousActorCriticModel, self).__init__()

        # initialize the spaces
        self._num_obs = num_observations
        self._num_actions = num_actions

        # create the actor and critic
        self._initializer = initializer = HeNormal()
        self.actor = self._create_actor(actor_hidden_units, actor_activation_function)
        self.critic = self._create_critic(critic_hidden_units, critic_activation_function)

        self._scales = np.ones((self._num_obs,), dtype=np.float32)

    def __call__(self, state_batch):
        state_batch = self._scale_network_input(state_batch)
        mu_batch, sigma_batch = self.actor(state_batch)
        value_estimate_batch = self.critic(state_batch)
        return {'mu': tf.squeeze(mu_batch), 'sigma': tf.squeeze(sigma_batch), 'value_estimate': tf.squeeze(value_estimate_batch)}

    def save(self, save_path):
        saving_path_actor = Path(save_path + '_actor')
        saving_path_actor.mkdir(parents=True, exist_ok=True)
        self.actor.save(saving_path_actor)

        saving_path_critic = Path(save_path + '_critic')
        saving_path_critic.mkdir(parents=True, exist_ok=True)
        self.critic.save(saving_path_critic)

    def load_model(self, load_path):
        directories = [x.split("_") for x in os.listdir(load_path)]
        directories = [[x[0], int(x[1]), x[2]] for x in directories]
        directory = sorted(directories, key=lambda x: x[1], reverse=True)[0]
        actor_directory = load_path + "\\epoch_" + str(directory[1]) + "_actor"
        critic_directory = load_path + "\\epoch_" + str(directory[1]) + "_critic"

        self.actor = load_model(actor_directory)
        self.critic = load_model(critic_directory)

    def _create_actor(self, actor_hidden_units, actor_activation_function):
        state_input = Input(shape=self._num_obs)

        # create the hidden layers
        next_input = state_input
        for i in range(len(actor_hidden_units)):
            units = actor_hidden_units[i]
            activation = actor_activation_function
            name = f'actor_dense_{i}'
            next_input = Dense(units, activation=activation, name=name, kernel_initializer=self._initializer)(next_input)

        # create the output layers
        mu = Dense(self._num_actions, activation='tanh', name='actor_mu', kernel_initializer=self._initializer)(next_input)
        sigma = tf.exp(Dense(self._num_actions, activation=None, name='actor_sigma', kernel_initializer=self._initializer)(next_input)) + SMALL_CONST
        return Model(inputs=state_input, outputs=[mu, sigma])

    def _create_critic(self, critic_hidden_units, critic_activation_function):
        state_input = Input(shape=self._num_obs)

        # create the hidden layers
        next_input = state_input
        for i in range(len(critic_hidden_units)):
            units = critic_hidden_units[i]
            activation = critic_activation_function
            name = f'critic_dense_{i}'
            next_input = Dense(units, activation=activation, name=name, kernel_initializer=self._initializer)(next_input)

        # create the output layer
        value_estimate = Dense(1, name='critic_value_estimate', kernel_initializer=self._initializer)(next_input)
        return Model(inputs=state_input, outputs=value_estimate)

    def _scale_network_input(self, input):
        """Scales the input with the currently highest received absolute input."""
        if tf.is_tensor(input):
            input = input.numpy()
        self._update_scales(input)
        return input / self._scales

    def _update_scales(self, input):
        """Updates the scales"""
        batch_max_vals = np.abs(input).max(axis=0)
        self._scales[batch_max_vals > self._scales] = batch_max_vals[batch_max_vals > self._scales]
