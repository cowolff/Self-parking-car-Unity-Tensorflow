import time

from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

import gym

from models import ContinuousActorCriticModel
from policies import ContinuousPolicy
from trainer import PPOTrainer


def main():

    config_channel = EngineConfigurationChannel()
    config_channel.set_configuration_parameters(width=1800, height=900, time_scale=10.0)

    # create the environment
    env_directory = "C:\\Users\\wolff\Desktop\\MLAgents\\envs_continuous\\DeepReinforcementLearning"
    unity_env = UnityEnvironment(env_directory, side_channels=[config_channel], no_graphics=False)
    env = UnityToGymWrapper(unity_env, allow_multiple_obs=False)

    # env = gym.make("LunarLanderContinuous-v2")
    # create the model
    model = ContinuousActorCriticModel(env.observation_space.shape[0], env.action_space.shape[0])
    model.load_model("C:\\Users\\wolff\\Documents\\GitHub\\Deep-Reinforcement-Learning-Self-driving-cars-in-unity\\saved_models\\1619383776")

    # create the policy
    policy = ContinuousPolicy(model)

    # create the trainer
    trainer = PPOTrainer(policy, env, learning_rate=0.0001, learning_rate_critic=0.0001, episodes=30000, sample_size=20480, batch_size=4096, gamma=0.99, clipping_value=0.2)
    trainer.set_saving_params(f'\\saved_models\\{int(time.time())}', 50, True)

    trainer.train()


if __name__ == '__main__':
    main()