import numpy as np
import tensorflow as tf


class Buffer:

    def __init__(self, capacity, batch_size, num_states, num_actions):
        self.capacity = capacity
        self.batch_size = batch_size
        self.current_index = 0

        self.state_buffer = np.zeros(shape=(self.capacity, num_states), dtype=np.float32)
        self.action_buffer = np.zeros(shape=(self.capacity, num_actions), dtype=np.float32)
        self.next_state_buffer = np.zeros(shape=(self.capacity, num_states), dtype=np.float32)
        self.reward_buffer = np.zeros(shape=(self.capacity,), dtype=np.float32)
        self.not_done_buffer = np.zeros(shape=(self.capacity,), dtype=np.float32)
        self.log_prob_buffer = np.zeros(shape=(self.capacity, num_actions), dtype=np.float32)
        self.value_estimate_buffer = np.zeros(shape=(self.capacity,), dtype=np.float32)

    def add(self, state, action, reward, next_state, not_done, log_prob, value_estimate):
        """Adds a new sample to the buffer."""
        index = self.current_index % self.capacity
        self.state_buffer[index] = state
        self.action_buffer[index] = action
        self.next_state_buffer[index] = next_state
        self.reward_buffer[index] = reward
        self.not_done_buffer[index] = not_done
        self.log_prob_buffer[index] = log_prob
        self.value_estimate_buffer[index] = value_estimate

        self.current_index = self.current_index + 1

    def reset(self):
        """Resets the buffer."""
        self.state_buffer = np.zeros_like(self.state_buffer, dtype=np.float32)
        self.action_buffer = np.zeros_like(self.action_buffer, dtype=np.float32)
        self.next_state_buffer = np.zeros_like(self.next_state_buffer, dtype=np.float32)
        self.reward_buffer = np.zeros_like(self.reward_buffer, dtype=np.float32)
        self.not_done_buffer = np.zeros_like(self.not_done_buffer, dtype=np.float32)
        self.log_prob_buffer = np.zeros_like(self.log_prob_buffer, dtype=np.float32)
        self.value_estimate_buffer = np.zeros_like(self.value_estimate_buffer, dtype=np.float32)

        self.current_index = 0

    def get_dataset(self):
        """Creates the dataset with all available data and returns it."""
        index = min(self.capacity, self.current_index)
        dataset = tf.data.Dataset.from_tensor_slices((self.state_buffer[:index],
                                                      self.action_buffer[:index],
                                                      self.next_state_buffer[:index],
                                                      self.reward_buffer[:index],
                                                      self.not_done_buffer[:index],
                                                      self.log_prob_buffer[:index],
                                                      self.value_estimate_buffer[:index]))
        return dataset.batch(self.batch_size, True)
