"""Sample script for training a control policy on the Hopper environment

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between TRPO, PPO, and SAC.
"""
import gym
from env.custom_hopper import *
from stable_baselines3 import PPO, SAC
from sb3_contrib import TRPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

def main():
    source_env = gym.make('CustomHopper-source-v0')
    target_env = gym.make('CustomHopper-target-v0')

    ############ STEP 2 ############

    episodes = 50

    #PPO Policy
    modelPPO = PPO("MlpPolicy", source_env, verbose=1)
    #modelPPO = PPO("MlpPolicy", source_env, verbose=1, learning_rate=0.0001)
    modelPPO.learn(total_timesteps=150_000)
    modelPPO.save("./PPO/PPO_source_3e-04_150K")
    #modelPPO = PPO.load("./PPO/PPO_source_3e-04_150K")
    mean_reward, std_reward = evaluate_policy(modelPPO, Monitor(target_env), episodes, render=False)
    print(f"Mean reward on target environment using PPO in 150K ts: {mean_reward:.2f} +/- {std_reward:.2f}")

    modelPPO = PPO("MlpPolicy", target_env, verbose=1)
    modelPPO.learn(total_timesteps=150_000)
    modelPPO.save("./PPO/PPO_target_3e-04_150K")
    #modelPPO = PPO.load("./PPO/./PPO/PPO_target_3e-04_150K")
    mean_reward, std_reward = evaluate_policy(modelPPO, Monitor(target_env), episodes, render=False)
    print(f"Mean reward on target environment using PPO in 150K ts: {mean_reward:.2f} +/- {std_reward:.2f}")
   
    #TRPO Policy
    modelTRPO = TRPO("MlpPolicy", source_env, verbose=1)
    #modelTRPO = TRPO("MlpPolicy", source_env, verbose=1, learning_rate=0.0003)
    modelTRPO.learn(total_timesteps=300_000) #, log_interval=4
    modelTRPO.save(f"./TRPO/TRPO_source_1e-03_300K") 
    mean_reward, std_reward = evaluate_policy(modelTRPO, Monitor(target_env), episodes, render=False)
    print(f"Mean reward after 300K timesteps on target environment using TRPO: {mean_reward:.2f} +/- {std_reward:.2f}")

    modelTRPO = TRPO("MlpPolicy", target_env, verbose=1)
    modelTRPO.learn(total_timesteps=500_000)
    modelTRPO.save(f"./TRPO/TRPO_target _1-e03_500K") 
    mean_reward, std_reward = evaluate_policy(modelTRPO, Monitor(target_env), episodes, render=False)
    print(f"Mean reward after 500K timesteps on target environment using TRPO: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    #SAC Policy
    modelSAC = SAC("MlpPolicy", source_env, verbose=1)
    #modelSAC = SAC("MlpPolicy", source_env, verbose=1, learning_rate=0.0003)
    modelSAC.learn(total_timesteps=25_000, log_interval=4)
    modelSAC.save("./SAC/SAC_source_3e-04_25K") 
    mean_reward, std_reward = evaluate_policy(modelSAC, Monitor(source_env), episodes, render=False)
    print(f"Mean reward on source environment using SAC: {mean_reward:.2f} +/- {std_reward:.2f}")

    mean_reward, std_reward = evaluate_policy(modelSAC, Monitor(target_env), episodes, render=False)
    print(f"Mean reward on target environment using SAC: {mean_reward:.2f} +/- {std_reward:.2f}") 

    ############ DOMAIN RANDOMIZATION - STEP 3 ############

    #### training phase
    source_env.set_udr_flag(True, 30)
    #TRPO Policy
    modelTRPO = TRPO("MlpPolicy", source_env)
    #modelTRPO = TRPO("MlpPolicy", source_env, verbose=1, learning_rate=0.0003)
    modelTRPO.learn(total_timesteps=200_000, progress_bar=True)
    source_env.set_udr_flag(False) 
    modelTRPO.save(f"./TRPO/TRPO_udr_30_1e-03_200K")
    #modelTRPO = TRPO.load("./TRPO/TRPO_udr_30_1e-03_200K")

    #### test phase
    mean_reward, std_reward = evaluate_policy(modelTRPO, Monitor(source_env), episodes, render=False)
    print(f"\nMean reward after 200K timesteps and 30% on source after UDR using TRPO: {mean_reward:.2f} +/- {std_reward:.2f}")
    mean_reward, std_reward = evaluate_policy(modelTRPO, Monitor(target_env), episodes, render=False)
    print(f"\nMean reward after 200K timesteps and 30% on target after UDR using TRPO: {mean_reward:.2f} +/- {std_reward:.2f}")
    source_env.reset_masses_ranges()

    #PPO Policy
    source_env.set_udr_flag(True, 40)
    modelPPO = PPO("MlpPolicy", source_env)
    modelPPO.learn(total_timesteps=500_000, progress_bar=True)
    source_env.set_udr_flag(False)
    modelPPO.save("./PPO/PPO_udr_40_3e-04_500K") 
    #modelPPO = PPO.load("./PPO/PPO_udr_40_3e-04_500K")
    
    #### test phase
    mean_reward, std_reward = evaluate_policy(modelPPO, Monitor(source_env), episodes, render=False)
    print(f"\nMean reward after 500K timesteps and 40% on source after UDR using PPO: {mean_reward:.2f} +/- {std_reward:.2f}")
    mean_reward, std_reward = evaluate_policy(modelPPO, Monitor(target_env), episodes, render=False)
    print(f"\nMean reward after 500K timesteps and 40% on target after UDR using PPO: {mean_reward:.2f} +/- {std_reward:.2f}")
    source_env.reset_masses_ranges()
    
    source_env.close()
    target_env.close()

    

if __name__ == '__main__':
    main()