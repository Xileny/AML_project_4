"""Sample script for training a control policy on the Hopper environment

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between TRPO, PPO, and SAC.
"""
import gym
from env.custom_hopper import *
from stable_baselines3 import PPO, SAC
from sb3_contrib import TRPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

def main():
    env = gym.make('CustomHopper-source-v0')
    #env2 = gym.make('CustomHopper-target-v0')
    target_env = gym.make('CustomHopper-target-v0')
    #check_env(env)

    """ print('State space:', env.observation_space)  # state-space
    print('Action space:', env.action_space)  # action-space
    print('Dynamics parameters:', env.get_parameters())  # masses of each link of the Hopper """

    """  print('Target state space:', env2.observation_space)  # state-space
    print('Target action space:', env2.action_space)  # action-space """
    #print('Target dynamics parameters:', env2.get_parameters())  # masses of each link of the Hopper

    #env.set_random_parameters()
    #print('Dynamics parameters 2:', env.get_parameters())  # masses of each link of the Hopper

    """
        TODO:

            - train a policy with stable-baselines3 on source env
            - test the policy with stable-baselines3 on <source,target> envs
    """
    """ #PPO Policy
    modelPPO = PPO("MlpPolicy", env, verbose=1, learning_rate=0.003)
    modelPPO.learn(total_timesteps=100_000)
    modelPPO.save("PPO lr_1e-3 100K timesteps") """

    """ #TRPO Policy
    modelTRPO = TRPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)
    modelTRPO.learn(total_timesteps=10_000, log_interval=4)
    modelTRPO.save("TRPO lr_3e-4 10K timesteps") """
    
    """ #SAC Policy
    modelSAC = SAC("MlpPolicy", env, verbose=1)
    modelSAC.learn(total_timesteps=25000, log_interval=4)
    modelSAC.save("SAC 25K timesteps") """

    modelPPO = PPO.load("./PPO/PPO 50K timesteps")
    env1 = Monitor(env)
    # Test the policy on the source env
    mean_reward, std_reward = evaluate_policy(modelPPO, env1)
    print(f"Mean reward on source environment using PPO: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Initialize the total reward and the episode counter
    total_reward = 0
    n_episodes = 0
    # Initialize the test environment state
    state = env.reset()
    #state = target_env.reset()
    while n_episodes < 50:
        # Execute an action 
        action, _states = modelPPO.predict(state)
        state, reward, done, _info = env.step(action)
        env.render()
        # Add the obtained reward to the total reward
        total_reward += reward
        # Increment the episode counter
        #print(f"Episode {n_episodes}, obtained reward: {reward}")
        n_episodes += 1
        # If the episode is done, re-initialize the test environment state
        if done:
            state = env.reset()

    #Compute the average reward
    average_reward = total_reward / n_episodes
    print(f"Average reward over {n_episodes} episodes: {average_reward:.2f}")

    """ modelTRPO = TRPO.load("TRPO lr_3e-4 10K timesteps")
    env2 = Monitor(env)
    # Testa la policy sull'ambiente sorgente
    mean_reward, std_reward = evaluate_policy(modelTRPO, env2)
    print(f"Mean reward on source environment using TRPO: {mean_reward:.2f} +/- {std_reward:.2f}")

    obs = env.reset()
    done = False
    total_reward = 0
    while True:
        action, _states = modelTRPO.predict(obs)
        obs, rewards, done, info = env.step(action)
        total_reward += rewards
        env.render() """

    """ modelSAC = SAC.load("SAC 25K timesteps")
    env3 = Monitor(env)
    # Testa la policy sull'ambiente sorgente
    mean_reward, std_reward = evaluate_policy(modelSAC, env3)
    print(f"Mean reward on source environment using SAC: {mean_reward:.2f} +/- {std_reward:.2f}") """    

    """ obs = env.reset()
    while True:
        action, _states = modelSAC.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render() """

    """  vec_env = model.get_env()
    print("vec_env: ", vec_env)
    obs = vec_env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render()
        # VecEnv resets automatically
        if done:
          obs = env.reset()

    env.close() """

if __name__ == '__main__':
    main()