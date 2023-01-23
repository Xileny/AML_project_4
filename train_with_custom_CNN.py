import gym
from env.custom_hopper import *
from sb3_contrib import TRPO
from stable_baselines3.common.callbacks import CheckpointCallback
from wrappers.pixel_observation_wrapper  import PixelObservationWrapper 
from wrappers.image_to_pytorch import ImageToPyTorch
from wrappers.stackframes import StackFrames
from wrappers.resize import  Resize
from wrappers.grayscale import Grayscale
from customCNN import CustomCNN

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

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
)

# Save a checkpoint every x steps
checkpoint_callback = CheckpointCallback(
  save_freq=25_000,
  save_path="./models_trained_with_images/",
  name_prefix="TRPO_udr_30_1e-02",
  save_replay_buffer=True,
  save_vecnormalize=True,
)

source_env.set_udr_flag(True, 30) 
model = TRPO("MlpPolicy", image_env, policy_kwargs=policy_kwargs, verbose=1, batch_size=32, device="cpu", learning_rate=0.01)
trained_model = model.learn(total_timesteps=150_000, progress_bar=True, callback=checkpoint_callback)
source_env.set_udr_flag(False)
trained_model.save(f"./models_trained_with_images/TRPO_udr_30_1e-02_150K") 
source_env.reset_masses_ranges()
