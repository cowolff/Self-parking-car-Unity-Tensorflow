from abc import ABC, abstractmethod

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model


class Policy(ABC):
    """The base policy from which all other policies inherit."""

    def __init__(self, model: Model):
        self._model = model

    def get_model(self):
        return self._model

    def get_value_estimate(self, state):
        """Returns the estimated value."""
        return self._model(state)['value_estimate']

    def save_model(self, saving_path):
        """Saves the model."""
        self._model.save(saving_path)

    def load_model(self, loading_path):
        """Loads the model from the given path."""
        self._model = tf.keras.models.load_model(loading_path)

    def get_trainable_variables(self, name=''):
        """Returns the trainable parameters of a network filtered with the given name."""
        all_trainable_vars = self._model.trainable_variables
        if name:
            return [var for var in all_trainable_vars if name in var.name]
        return all_trainable_vars

    @abstractmethod
    def get_action(self, state, return_log_prob=False, is_test=False):
        """Returns the action for the given state."""

    @abstractmethod
    def get_log_prob(self, state, action):
        """Returns the logarithmic probability of taking the given action in the given state."""


class ContinuousPolicy(Policy):
    """A continuous policy that can be trained by a trainer."""

    def __init__(self, model):
        super(ContinuousPolicy, self).__init__(model)

    def get_action(self, state, return_log_prob=False, is_test=False):
        network_out = self._model(state)
        mu, sigma = network_out["mu"], network_out["sigma"]
        norm_dist = tfp.distributions.Normal(mu, sigma)
        action = norm_dist.sample()
        output = {"action": action.numpy()}

        if return_log_prob:
            output["log_prob"] = norm_dist.log_prob(action).numpy()

        value_estimate_key = 'value_estimate'
        if value_estimate_key in network_out:
            output[value_estimate_key] = network_out[value_estimate_key].numpy()

        return output

    def get_log_prob(self, state, action):
        network_out = self._model(state)
        mu, sigma = network_out["mu"], network_out["sigma"]
        norm_dist = tfp.distributions.Normal(mu, sigma)

        return norm_dist.log_prob(action)
