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
    source_env = gym.make('CustomHopper-source-v0')
    target_env = gym.make('CustomHopper-target-v0')
    #check_env(env)

    """
        TODO:

            - train a policy with stable-baselines3 on source env
            - test the policy with stable-baselines3 on <source,target> envs
    """
    """ #PPO Policy
    modelPPO = PPO("MlpPolicy", source_env, verbose=1, learning_rate=0.003)
    modelPPO.learn(total_timesteps=100_000)
    modelPPO.save("PPO lr_1e-3 100K timesteps") """

    """ #TRPO Policy
    modelTRPO = TRPO("MlpPolicy", source_env, verbose=1, learning_rate=0.0003)
    modelTRPO.learn(total_timesteps=10_000, log_interval=4)
    modelTRPO.save("TRPO lr_3e-4 10K timesteps") """
    
    """ #SAC Policy
    modelSAC = SAC("MlpPolicy", source_env, verbose=1)
    modelSAC.learn(total_timesteps=25000, log_interval=4)
    modelSAC.save("SAC 25K timesteps") """

    episodes = 50

    modelPPO = PPO.load("./PPO/PPO 100K timesteps")
    # Test the policy on the source env
    mean_reward, std_reward = evaluate_policy(modelPPO, Monitor(source_env), episodes, render=True)
    print(f"Mean reward on source environment using PPO: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Test the policy on the target env
    mean_reward, std_reward = evaluate_policy(modelPPO, Monitor(target_env), episodes, render=False)
    print(f"Mean reward on target environment using PPO: {mean_reward:.2f} +/- {std_reward:.2f}")

    """ modelTRPO = TRPO.load("TRPO lr_3e-4 10K timesteps")
    # Testa la policy sull'ambiente sorgente
    mean_reward, std_reward = evaluate_policy(modelTRPO, Monitor(source_env))
    print(f"Mean reward on source environment using TRPO: {mean_reward:.2f} +/- {std_reward:.2f}") """


    """ modelSAC = SAC.load("SAC 25K timesteps")
    # Testa la policy sull'ambiente sorgente
    mean_reward, std_reward = evaluate_policy(modelSAC, Monitor(source_env))
    print(f"Mean reward on source environment using SAC: {mean_reward:.2f} +/- {std_reward:.2f}") """    

   
    source_env.close()
    target_env.close()

    

if __name__ == '__main__':
    main()