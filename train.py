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

    episodes = 50
    timesteps = 500_000

    """
        TODO:

            - train a policy with stable-baselines3 on source env
            - test the policy with stable-baselines3 on <source,target> envs
    """
    ############ TRAINING PHASE ############

    #PPO Policy
    modelPPO = PPO("MlpPolicy", target_env, verbose=1)
    #modelPPO = PPO("MlpPolicy", source_env, verbose=1, learning_rate=0.0001)
    modelPPO.learn(total_timesteps=150_000)
    modelPPO.save("./PPO/PPO train_on_target 150K timesteps")
    #modelPPO = PPO.load("./PPO/PPO lr_1e-4 150K timesteps")
    mean_reward, std_reward = evaluate_policy(modelPPO, Monitor(target_env), episodes, render=False)
    print(f"Mean reward on target environment using PPO in 150K ts: {mean_reward:.2f} +/- {std_reward:.2f}")

    modelPPO = PPO("MlpPolicy", target_env, verbose=1)
    #modelPPO = PPO("MlpPolicy", source_env, verbose=1, learning_rate=0.0001)
    modelPPO.learn(total_timesteps=175_000)
    modelPPO.save("./PPO/PPO train_on_target 175K timesteps")
    #modelPPO = PPO.load("./PPO/PPO lr_1e-4 150K timesteps")
    mean_reward, std_reward = evaluate_policy(modelPPO, Monitor(target_env), episodes, render=False)
    print(f"Mean reward on target environment using PPO in 175K ts: {mean_reward:.2f} +/- {std_reward:.2f}")
   
    #TRPO Policy
    modelTRPO = TRPO("MlpPolicy", target_env, verbose=1)
    #modelTRPO = TRPO("MlpPolicy", source_env, verbose=1, learning_rate=0.0003)
    modelTRPO.learn(total_timesteps=300_000) #, log_interval=4
    modelTRPO.save(f"./TRPO/TRPO train_on_target 300K timesteps") 
    mean_reward, std_reward = evaluate_policy(modelTRPO, Monitor(target_env), episodes, render=False)
    print(f"Mean reward after 300K timesteps on target environment using TRPO: {mean_reward:.2f} +/- {std_reward:.2f}")

    modelTRPO = TRPO("MlpPolicy", target_env, verbose=1)
    #modelTRPO = TRPO("MlpPolicy", source_env, verbose=1, learning_rate=0.0003)
    modelTRPO.learn(total_timesteps=500_000) #, log_interval=4
    modelTRPO.save(f"./TRPO/TRPO train_on_target 500K timesteps") 
    mean_reward, std_reward = evaluate_policy(modelTRPO, Monitor(target_env), episodes, render=False)
    print(f"Mean reward after 500K timesteps on target environment using TRPO: {mean_reward:.2f} +/- {std_reward:.2f}")

    modelTRPO = TRPO("MlpPolicy", target_env, verbose=1)
    #modelTRPO = TRPO("MlpPolicy", source_env, verbose=1, learning_rate=0.0003)
    modelTRPO.learn(total_timesteps=200_000) #, log_interval=4
    modelTRPO.save(f"./TRPO/TRPO train_on_target 200K timesteps") 
    mean_reward, std_reward = evaluate_policy(modelTRPO, Monitor(target_env), episodes, render=False)
    print(f"Mean reward after 200K timesteps on target environment using TRPO: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    """ #SAC Policy
    modelSAC = SAC("MlpPolicy", source_env, verbose=1)
    #modelSAC = SAC("MlpPolicy", source_env, verbose=1, learning_rate=0.0003)
    modelSAC.learn(total_timesteps=25000, log_interval=4)
    modelSAC.save("./SAC/SAC 25K timesteps") 
    mean_reward, std_reward = evaluate_policy(modelSAC, Monitor(source_env), episodes, render=False)
    print(f"Mean reward on source environment using SAC: {mean_reward:.2f} +/- {std_reward:.2f}")"""

    ############ END OF THE TRAINING PHASE ############

    ############ TEST PHASE ############

    """ modelPPO = PPO.load("./PPO/PPO 150K timesteps")
    # Test the policy on the source env
    mean_reward, std_reward = evaluate_policy(modelPPO, Monitor(source_env), episodes, render=False)
    print(f"Mean reward on source environment using PPO: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Test the policy on the target env
    mean_reward, std_reward = evaluate_policy(modelPPO, Monitor(target_env), episodes, render=False)
    print(f"Mean reward on target environment using PPO in 150K ts: {mean_reward:.2f} +/- {std_reward:.2f}")

    modelPPO = PPO.load("./PPO/PPO 175K timesteps")
    # Test the policy on the target env
    mean_reward, std_reward = evaluate_policy(modelPPO, Monitor(target_env), episodes, render=False)
    print(f"Mean reward on target environment using PPO in 175K ts: {mean_reward:.2f} +/- {std_reward:.2f}") """

    """ modelSAC = SAC.load("SAC 25K timesteps")
    # Testa la policy sull'ambiente sorgente
    mean_reward, std_reward = evaluate_policy(modelSAC, Monitor(source_env))
    print(f"Mean reward on source environment using SAC: {mean_reward:.2f} +/- {std_reward:.2f}") """    

    ############ END OF THE TEST PHASE ############
   
    source_env.close()
    target_env.close()

    

if __name__ == '__main__':
    main()