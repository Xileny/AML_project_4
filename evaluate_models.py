from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import TRPO
from wrappers.pixel_observation_wrapper  import PixelObservationWrapper 
from wrappers.image_to_pytorch import ImageToPyTorch
from wrappers.stackframes import StackFrames
from wrappers.resize import  Resize
from wrappers.grayscale import Grayscale
import gym
from env.custom_hopper import *

source_env = gym.make('CustomHopper-source-v0')
target_env = gym.make('CustomHopper-target-v0')

pixel_env = PixelObservationWrapper(source_env)
grayscale_env = Grayscale(pixel_env)
resized_env = Resize(grayscale_env, 100, 100)
stackframes_env = StackFrames(resized_env, 4)
image_env = ImageToPyTorch(stackframes_env)

pixel_target_env = PixelObservationWrapper(target_env)
grayscale_target_env = Grayscale(pixel_target_env)
resized_target_env = Resize(grayscale_target_env, 100, 100)
stackframes_target_env = StackFrames(resized_target_env, 4)
image_target_env = ImageToPyTorch(stackframes_target_env)

trained_model = TRPO.load("./models_trained_with_images/TRPO_udr_30_1e-02_150K")
mean_reward, std_reward = evaluate_policy(trained_model, Monitor(image_env), 50)
print(f"\nMean reward on source environment : {mean_reward:.2f} +/- {std_reward:.2f}")
mean_reward, std_reward = evaluate_policy(trained_model, Monitor(image_target_env), 50)
print(f"\nMean reward on target environment : {mean_reward:.2f} +/- {std_reward:.2f}")
