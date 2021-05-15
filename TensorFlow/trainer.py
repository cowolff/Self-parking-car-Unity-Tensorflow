import os
from abc import ABC, abstractmethod
from pathlib import Path

from policies import Policy
from buffer import Buffer

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MSE
import csv
import time
from keras import backend as K

from gym import Env


class Trainer(ABC):
    """Base trainer that optimizes a policy."""

    def __init__(self, policy: Policy, env: Env, learning_rate: float = 0.001, episodes: int = 1e5):
        self._policy = policy
        self._env = env
        self._optimizer = Adam(learning_rate)
        self._episodes = episodes

        self._path_to_saved_model = None
        self._path_to_saving_dir = None
        self._saving_episodes = 0
        self._save_intermediate_model = False
        self._save_final_model = False
        self._log_file = f"log{int(time.time())}.csv"


    def set_loading_params(self, path_to_saved_model):
        """Sets the path to the model that should be loaded."""
        self._path_to_saved_model = path_to_saved_model


    def set_saving_params(self, path_to_saving_dir: str, saving_episodes: int = 10, save_final_model: bool = True):
        """Sets the parameters to save the model."""
        self._saving_episodes = saving_episodes
        self._save_intermediate_model = True
        self._save_final_model = save_final_model

        self._saving_path = Path(os.getcwd() + path_to_saving_dir)
        self._saving_path.mkdir(parents=True, exist_ok=True)


    def train(self):
        """Training loop."""
        # load the last model if specified
        self._load_model()
        try:
            for episode in range(1, self._episodes):
                self._train_one_episode()

                # save intermediate model
                if self._save_intermediate_model and episode % self._saving_episodes == 0:
                    saving_path = os.path.join(str(self._saving_path), f'epoch_{episode}')
                    self._save_model(saving_path)
        except:
            if self._save_intermediate_model:
                saving_path = os.path.join(str(self._saving_path), f'epoch_{episode}')
                self._save_model(saving_path)
            exit(0)

        if self._save_final_model:
            saving_path = os.path.join(str(self._saving_path), 'final')
            self._save_model(saving_path)

    def _load_model(self):
        """Load the model from the path."""
        if self._path_to_saved_model is not None:
            self._policy.load_model(self._path_to_saved_model)

    def _save_model(self, saving_path):
        """Saves the model."""
        self._policy.save_model(saving_path)

    @abstractmethod
    def _train_one_episode(self):
        """Trains the policy for one epoch."""


class PPOTrainer(Trainer):
    """Trains a policy with the PPO algorithm."""

    def __init__(self, policy: Policy, env: Env, learning_rate: float = 0.001, learning_rate_critic: float = 0.001, episodes: int = 1e5, sample_size: int = 1e5, batch_size: int = 512, gamma: float = 0.99, clipping_value: float = 0.2):
        super(PPOTrainer, self).__init__(policy, env, learning_rate, episodes)

        self._actor_optimizer = self._optimizer
        self._critic_optimizer = Adam(learning_rate_critic)
        self._sample_size = sample_size
        self._gamma = gamma
        self._clipping_value = clipping_value

        self._buffer = Buffer(self._sample_size, batch_size, self._env.observation_space.shape[0], self._env.action_space.shape[0])

        self._actor_trainable_vars = self._policy.get_trainable_variables('actor')
        self._critic_trainable_vars = self._policy.get_trainable_variables('critic')

    def _train_one_episode(self):
        batches = self._collect_data()

        actor_losses = []
        critic_losses = []
        rewards = []
        not_dones = []
        for state, action, next_state, reward, not_done, log_prob, value_estimate in batches:
            value = reward + not_done * self._gamma * self._policy.get_value_estimate(next_state)
            advantage = value - value_estimate

            actor_loss = self._update_actor(state, action, log_prob, advantage)
            critic_loss = self._update_critic(state, value)

            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            rewards.append(reward)
            not_dones.append(not_done)
        print("Reward:", sum(sum(rewards).numpy()))
        self._log_data(rewards, not_dones)
        
    def _log_data(self, rewards, not_dones):
        try:
            with open(self._log_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([str(int(sum(sum(rewards).numpy())))])
        except FileExistsError:
            with open(self._log_file, 'w+', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([str(int(sum(sum(rewards).numpy())))])

    def _collect_data(self):
        self._buffer.reset()
        state = np.reshape(self._env.reset(), (1, -1))
        for i in range(self._sample_size):
            #self._env.render()
            output = self._policy.get_action(state, True)
            next_state, reward, done, _ = self._env.step(output['action'])

            if done:
                next_state = self._env.reset()
            self._buffer.add(state, output['action'], reward, next_state, not done, output['log_prob'], output['value_estimate'])
            state = np.reshape(next_state, (1, -1))
        return self._buffer.get_dataset()

    def _update_actor(self, state, action, log_prob, advantage, noise=1.0):
        """Trains the policy network with PPO clipped loss."""

        with tf.GradientTape() as tape:
            # calculate the probability ratio
            new_log_prob = self._policy.get_log_prob(state, action)

            log_prob = tf.cast(log_prob, dtype=tf.float32)

            prob_ratio = tf.exp(new_log_prob - log_prob)

            # calculate the loss - PPO
            unclipped_loss = prob_ratio * tf.expand_dims(advantage, 1)
            clipped_loss = tf.clip_by_value(prob_ratio, 1 - self._clipping_value, 1 + self._clipping_value) * tf.expand_dims(
                advantage, 1)
            loss = -tf.reduce_mean(tf.minimum(unclipped_loss, clipped_loss))
            gradients = tape.gradient(loss, self._policy.get_trainable_variables('actor'))

        self._actor_optimizer.apply_gradients(zip(gradients, self._policy.get_trainable_variables('actor')))
        return loss

    def _update_critic(self, state, value):
        """Trains the value network with the mean squared error between the true and estimated value."""

        with tf.GradientTape() as tape:
            value_pred = self._policy.get_value_estimate(state)
            loss = MSE(value, value_pred)
            gradients = tape.gradient(loss, self._policy.get_trainable_variables('critic'))

        self._critic_optimizer.apply_gradients(zip(gradients, self._policy.get_trainable_variables('critic')))
        return loss
