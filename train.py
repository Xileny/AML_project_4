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

    """
        TODO:

            - train a policy with stable-baselines3 on source env
            - test the policy with stable-baselines3 on <source,target> envs
    """
    ############ STEP 2 ############
    episodes = 50
    timesteps = 500_000
    ############ TRAINING PHASE ############

    """ #PPO Policy
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
    print(f"Mean reward after 200K timesteps on target environment using TRPO: {mean_reward:.2f} +/- {std_reward:.2f}") """
    
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

    ############ DOMAIN RANDOMIZATION - STEP 3 ############

    #### training phase
    source_env.set_udr_flag(True, 40)
    #TRPO Policy
    modelTRPO = TRPO("MlpPolicy", source_env)
    #modelTRPO = TRPO("MlpPolicy", source_env, verbose=1, learning_rate=0.0003)
    modelTRPO.learn(total_timesteps=200_000, progress_bar=True)
    source_env.set_udr_flag(False) 
    modelTRPO.save(f"./TRPO/TRPO with_udr 200K 40percentage timesteps")
    #modelTRPO = TRPO.load("./TRPO/TRPO with_udr 200K timesteps")

    #### test phase
    mean_reward, std_reward = evaluate_policy(modelTRPO, Monitor(source_env), episodes, render=False)
    print(f"\nMean reward after 200K timesteps on source environment after applying UDR during training using TRPO: {mean_reward:.2f} +/- {std_reward:.2f}")
    mean_reward, std_reward = evaluate_policy(modelTRPO, Monitor(target_env), episodes, render=False)
    print(f"\nMean reward after 200K timesteps on target environment after applying UDR during training using TRPO: {mean_reward:.2f} +/- {std_reward:.2f}")

    source_env.set_udr_flag(True, 10)
    #TRPO Policy
    #modelTRPO = TRPO("MlpPolicy", source_env)
    modelTRPO = TRPO("MlpPolicy", source_env, learning_rate=0.0003)
    modelTRPO.learn(total_timesteps=300_000, progress_bar=True)
    source_env.set_udr_flag(False) 
    modelTRPO.save(f"./TRPO/TRPO with_udr lr_3e-4 10percentage 300K timesteps")
    #modelTRPO = TRPO.load("./TRPO/TRPO with_udr 200K timesteps")

    #### test phase
    mean_reward, std_reward = evaluate_policy(modelTRPO, Monitor(source_env), episodes, render=False)
    print(f"\nMean reward after 300K timesteps on source environment after applying UDR during training using TRPO: {mean_reward:.2f} +/- {std_reward:.2f}")
    mean_reward, std_reward = evaluate_policy(modelTRPO, Monitor(target_env), episodes, render=False)
    print(f"\nMean reward after 300K timesteps on target environment after applying UDR during training using TRPO: {mean_reward:.2f} +/- {std_reward:.2f}")

    source_env.set_udr_flag(True, 20)
    #TRPO Policy
    #modelTRPO = TRPO("MlpPolicy", source_env)
    modelTRPO = TRPO("MlpPolicy", source_env, learning_rate=0.0003)
    modelTRPO.learn(total_timesteps=300_000, progress_bar=True)
    source_env.set_udr_flag(False) 
    modelTRPO.save(f"./TRPO/TRPO with_udr lr_3e-4 20percentage 300K timesteps")
    #modelTRPO = TRPO.load("./TRPO/TRPO with_udr 200K timesteps")

    #### test phase
    mean_reward, std_reward = evaluate_policy(modelTRPO, Monitor(source_env), episodes, render=False)
    print(f"\nMean reward after 300K timesteps on source environment after applying UDR during training using TRPO: {mean_reward:.2f} +/- {std_reward:.2f}")
    mean_reward, std_reward = evaluate_policy(modelTRPO, Monitor(target_env), episodes, render=False)
    print(f"\nMean reward after 300K timesteps on target environment after applying UDR during training using TRPO: {mean_reward:.2f} +/- {std_reward:.2f}")

    source_env.set_udr_flag(True, 30)
    #TRPO Policy
    #modelTRPO = TRPO("MlpPolicy", source_env)
    modelTRPO = TRPO("MlpPolicy", source_env, learning_rate=0.0003)
    modelTRPO.learn(total_timesteps=300_000, progress_bar=True)
    source_env.set_udr_flag(False) 
    modelTRPO.save(f"./TRPO/TRPO with_udr lr_3e-4 30percentage 300K timesteps")
    #modelTRPO = TRPO.load("./TRPO/TRPO with_udr 200K timesteps")

    #### test phase
    mean_reward, std_reward = evaluate_policy(modelTRPO, Monitor(source_env), episodes, render=False)
    print(f"\nMean reward after 300K timesteps on source environment after applying UDR during training using TRPO: {mean_reward:.2f} +/- {std_reward:.2f}")
    mean_reward, std_reward = evaluate_policy(modelTRPO, Monitor(target_env), episodes, render=False)
    print(f"\nMean reward after 300K timesteps on target environment after applying UDR during training using TRPO: {mean_reward:.2f} +/- {std_reward:.2f}")

    source_env.set_udr_flag(True, 10)
    #PPO Policy
    modelPPO = PPO("MlpPolicy", source_env)
    #modelPPO = PPO("MlpPolicy", source_env, verbose=1, learning_rate=0.0001)
    modelPPO.learn(total_timesteps=175_000, progress_bar=True)
    source_env.set_udr_flag(False)
    modelPPO.save("./PPO/PPO with_udr 10percentage 175K timesteps") 
    #modelPPO = PPO.load("./PPO/PPO with_udr and uniform distrib 175K timesteps")
    
    #### test phase
    mean_reward, std_reward = evaluate_policy(modelPPO, Monitor(source_env), episodes, render=False)
    print(f"\nMean reward after 175K timesteps on source environment after applying UDR during training using PPO: {mean_reward:.2f} +/- {std_reward:.2f}")
    mean_reward, std_reward = evaluate_policy(modelPPO, Monitor(target_env), episodes, render=False)
    print(f"\nMean reward after 175K timesteps on target environment after applying UDR during training using PPO: {mean_reward:.2f} +/- {std_reward:.2f}")

    source_env.set_udr_flag(True, 20)
    #PPO Policy
    modelPPO = PPO("MlpPolicy", source_env)
    #modelPPO = PPO("MlpPolicy", source_env, verbose=1, learning_rate=0.0001)
    modelPPO.learn(total_timesteps=175_000, progress_bar=True)
    source_env.set_udr_flag(False)
    modelPPO.save("./PPO/PPO with_udr 20percentage 175K timesteps") 
    #modelPPO = PPO.load("./PPO/PPO with_udr and uniform distrib 175K timesteps")
    
    #### test phase
    mean_reward, std_reward = evaluate_policy(modelPPO, Monitor(source_env), episodes, render=False)
    print(f"\nMean reward after 175K timesteps on source environment after applying UDR during training using PPO: {mean_reward:.2f} +/- {std_reward:.2f}")
    mean_reward, std_reward = evaluate_policy(modelPPO, Monitor(target_env), episodes, render=False)
    print(f"\nMean reward after 175K timesteps on target environment after applying UDR during training using PPO: {mean_reward:.2f} +/- {std_reward:.2f}")

    source_env.set_udr_flag(True, 30)
    #PPO Policy
    modelPPO = PPO("MlpPolicy", source_env)
    #modelPPO = PPO("MlpPolicy", source_env, verbose=1, learning_rate=0.0001)
    modelPPO.learn(total_timesteps=175_000, progress_bar=True)
    source_env.set_udr_flag(False)
    modelPPO.save("./PPO/PPO with_udr 30percentage 175K timesteps") 
    #modelPPO = PPO.load("./PPO/PPO with_udr and uniform distrib 175K timesteps")
    
    #### test phase
    mean_reward, std_reward = evaluate_policy(modelPPO, Monitor(source_env), episodes, render=False)
    print(f"\nMean reward after 175K timesteps on source environment after applying UDR during training using PPO: {mean_reward:.2f} +/- {std_reward:.2f}")
    mean_reward, std_reward = evaluate_policy(modelPPO, Monitor(target_env), episodes, render=False)
    print(f"\nMean reward after 175K timesteps on target environment after applying UDR during training using PPO: {mean_reward:.2f} +/- {std_reward:.2f}")

    source_env.close()
    target_env.close()

    

if __name__ == '__main__':
    main()