import gym
from env.custom_hopper import *
from stable_baselines3 import PPO, SAC
from sb3_contrib import TRPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import os.path
from os import path
# from stable_baselines3.common.callbacks import EvalCallback
# from gym.wrappers.pixel_observation import PixelObservationWrapper
# from gym.wrappers.flatten_observation import FlattenObservation
from gym.wrappers.frame_stack import FrameStack
# from gym.wrappers.resize_observation import ResizeObservation
# from gym.wrappers.gray_scale_observation import GrayScaleObservation
import matplotlib.pyplot as plt
import torch as th
import torch.nn as nn
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym.wrappers.pixel_observation import (
    PixelObservationWrapper as gymWrapper)
from collections import deque
import numpy as np
import warnings
from skimage import color, img_as_ubyte
from skimage.transform import resize
import warnings
from stable_baselines3.common.callbacks import CheckpointCallback

source_env = gym.make('CustomHopper-source-v0')
target_env = gym.make('CustomHopper-target-v0')

"""Pixel observation wrapper for gym.Env."""


class PixelObservationWrapper(gym.Wrapper):
    """Pixel observation wrapper for obtaining pixel observations.
    Instead of returning the default environment observation, the wrapped
    environment's render function is used to produce RGB pixel observations.
    This behaves like gym.wrappers.PixelObservationWrapper but returns a
    gym.spaces.Box observation space and observation instead of
    a gym.spaces.Dict.
    Args:
        env (gym.Env): The environment to wrap. This environment must produce
            non-pixel observations and have a Box observation space.
        headless (bool): If true, this creates a window to init GLFW. Set to
            true if running on a headless machine or with a dummy X server,
            false otherwise.
    """

    def __init__(self, env, headless=True):
        if headless:
            # pylint: disable=import-outside-toplevel
            # this import fails without a valid mujoco license
            # so keep this here to avoid unecessarily requiring
            # a mujoco license everytime the wrappers package is
            # accessed.
            from mujoco_py import GlfwContext
            GlfwContext(offscreen=True)
        env.reset()
        env = gymWrapper(env)
        super().__init__(env)
        self._observation_space = env.observation_space['pixels']

    @property
    def observation_space(self):
        """gym.spaces.Box: Environment observation space."""
        return self._observation_space

    @observation_space.setter
    def observation_space(self, observation_space):
        self._observation_space = observation_space

    def reset(self, **kwargs):
        """gym.Env reset function.
        Args:
            kwargs (dict): Keyword arguments to be passed to gym.Env.reset.
        Returns:
            np.ndarray: Pixel observation of shape :math:`(O*, )`
                from the wrapped environment.
        """
        return self.env.reset(**kwargs)['pixels']

    def step(self, action):
        """gym.Env step function.
        Performs one action step in the enviornment.
        Args:
            action (np.ndarray): Action of shape :math:`(A*, )`
                to pass to the environment.
        Returns:
            np.ndarray: Pixel observation of shape :math:`(O*, )`
                from the wrapped environment.
            float : Amount of reward returned after previous action.
            bool : Whether the episode has ended, in which case further step()
                calls will return undefined results.
            dict: Contains auxiliary diagnostic information (helpful for
                debugging, and sometimes learning).
        """
        obs, reward, done, info = self.env.step(action)
        return obs['pixels'], reward, done, info

class ImageToPyTorch(gym.ObservationWrapper):
    """
    Image shape to num_channels x weight x height
    """

    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
            
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=255.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.uint8)

    def observation(self, observation):
        frame = np.swapaxes(observation, 2, 0)[None,:,:,:]
        
        return frame

"""Stack frames wrapper for gym.Env."""

class StackFrames(gym.Wrapper):
    """gym.Env wrapper to stack multiple frames.
    Useful for training feed-forward agents on dynamic games.
    Only works with gym.spaces.Box environment with 2D single channel frames.
    Args:
        env (gym.Env): gym.Env to wrap.
        n_frames (int): number of frames to stack.
        axis (int): Axis to stack frames on. This should be 2 for tensorflow
            and 0 for pytorch.
    Raises:
         ValueError: If observation space shape is not 2 dimnesional,
         if the environment is not gym.spaces.Box, or if the specified axis
         is not 0 or 2.
    """

    def __init__(self, env, n_frames, axis=2):
        if axis not in (0, 2):
            raise ValueError('Frame stacking axis should be 0 for pytorch or '
                             '2 for tensorflow.')
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise ValueError('Stack frames only works with gym.spaces.Box '
                             'environment.')

        if len(env.observation_space.shape) != 2:
            raise ValueError(
                'Stack frames only works with 2D single channel images')

        super().__init__(env)

        self._n_frames = n_frames
        self._axis = axis
        self._frames = deque(maxlen=n_frames)

        new_obs_space_shape = env.observation_space.shape + (n_frames, )
        if axis == 0:
            new_obs_space_shape = (n_frames, ) + env.observation_space.shape

        _low = env.observation_space.low.flatten()[0]
        _high = env.observation_space.high.flatten()[0]
        self._observation_space = gym.spaces.Box(
            _low,
            _high,
            shape=new_obs_space_shape,
            dtype=env.observation_space.dtype)

    @property
    def observation_space(self):
        """gym.spaces.Box: gym.Env observation space."""
        return self._observation_space

    @observation_space.setter
    def observation_space(self, observation_space):
        self._observation_space = observation_space

    def _stack_frames(self):
        """Stacks and returns the last n_frames.
        Returns:
            np.ndarray: stacked observation with shape either
            :math:`(N, n_frames, O*)` or :math:(N, O*, n_frames),
            depending on the axis specified.
        """
        return np.stack(self._frames, axis=self._axis)

    # pylint: disable=arguments-differ
    def reset(self):
        """gym.Env reset function.
        Returns:
            np.ndarray: Observation conforming to observation_space
            float: Reward for this step
            bool: Termination signal
            dict: Extra information from the environment.
        """
        observation = self.env.reset()
        self._frames.clear()
        for _ in range(self._n_frames):
            self._frames.append(observation)

        return self._stack_frames()

    def step(self, action):
        """gym.Env step function.
        Args:
            action (int): index of the action to take.
        Returns:
            np.ndarray: Observation conforming to observation_space
            float: Reward for this step
            bool: Termination signal
            dict: Extra information from the environment.
        """
        new_observation, reward, done, info = self.env.step(action)
        self._frames.append(new_observation)

        return self._stack_frames(), reward, done, info

"""Grayscale wrapper for gym.Env."""

class Grayscale(gym.Wrapper):
    """Grayscale wrapper for gym.Env, converting frames to grayscale.
    Only works with gym.spaces.Box environment with 2D RGB frames.
    The last dimension (RGB) of environment observation space will be removed.
    Example:
        env = gym.make('Env')
        # env.observation_space = (100, 100, 3)
        env_wrapped = Grayscale(gym.make('Env'))
        # env.observation_space = (100, 100)
    Args:
        env (gym.Env): Environment to wrap.
    Raises:
        ValueError: If observation space shape is not 3
            or environment is not gym.spaces.Box.
    """

    def __init__(self, env):
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise ValueError(
                'Grayscale only works with gym.spaces.Box environment.')

        if len(env.observation_space.shape) != 3:
            raise ValueError('Grayscale only works with 2D RGB images')

        super().__init__(env)

        _low = env.observation_space.low.flatten()[0]
        _high = env.observation_space.high.flatten()[0]
        assert _low == 0
        assert _high == 255
        self._observation_space = gym.spaces.Box(
            _low,
            _high,
            shape=env.observation_space.shape[:-1],
            dtype=np.uint8)

    @property
    def observation_space(self):
        """gym.Env: Observation space."""
        return self._observation_space

    @observation_space.setter
    def observation_space(self, observation_space):
        self._observation_space = observation_space

    def reset(self, **kwargs):
        """gym.Env reset function.
        Args:
            **kwargs: Unused.
        Returns:
            np.ndarray: Observation conforming to observation_space
        """
        del kwargs
        return _color_to_grayscale(self.env.reset())

    def step(self, action):
        """See gym.Env.
        Args:
            action (np.ndarray): Action conforming to action_space
        Returns:
            np.ndarray: Observation conforming to observation_space
            float: Reward for this step
            bool: Termination signal
            dict: Extra information from the environment.
        """
        obs, reward, done, info = self.env.step(action)
        return _color_to_grayscale(obs), reward, done, info


def _color_to_grayscale(obs):
    """Convert a 3-channel color observation image to grayscale and uint8.
    Args:
       obs (np.ndarray): Observation array, conforming to observation_space
    Returns:
       np.ndarray: 1-channel grayscale version of obs, represented as uint8
    """
    with warnings.catch_warnings():
        # Suppressing warning for possible precision loss when converting
        # from float64 to uint8
        warnings.simplefilter('ignore')
        return img_as_ubyte(color.rgb2gray((obs)))

"""Resize wrapper for gym.Env."""

class Resize(gym.Wrapper):
    """gym.Env wrapper for resizing frame to (width, height).
    Only works with gym.spaces.Box environment with 2D single channel frames.
    Example:
        | env = gym.make('Env')
        | # env.observation_space = (100, 100)
        | env_wrapped = Resize(gym.make('Env'), width=64, height=64)
        | # env.observation_space = (64, 64)
    Args:
        env: gym.Env to wrap.
        width: resized frame width.
        height: resized frame height.
    Raises:
        ValueError: If observation space shape is not 2
            or environment is not gym.spaces.Box.
    """

    def __init__(self, env, width, height):
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise ValueError('Resize only works with Box environment.')

        if len(env.observation_space.shape) != 2:
            raise ValueError('Resize only works with 2D single channel image.')

        super().__init__(env)

        _low = env.observation_space.low.flatten()[0]
        _high = env.observation_space.high.flatten()[0]
        self._dtype = env.observation_space.dtype
        self._observation_space = gym.spaces.Box(_low,
                                                 _high,
                                                 shape=[width, height],
                                                 dtype=self._dtype)

        self._width = width
        self._height = height

    @property
    def observation_space(self):
        """gym.Env observation space."""
        return self._observation_space

    @observation_space.setter
    def observation_space(self, observation_space):
        self._observation_space = observation_space

    def _observation(self, obs):
        with warnings.catch_warnings():
            """
            Suppressing warnings for
            1. possible precision loss when converting from float64 to uint8
            2. anti-aliasing will be enabled by default in skimage 0.15
            """
            warnings.simplefilter('ignore')
            obs = resize(obs, (self._width, self._height))  # now it's float
            if self._dtype == np.uint8:
                obs = img_as_ubyte(obs)
        return obs

    def reset(self):
        """gym.Env reset function."""
        return self._observation(self.env.reset())

    def step(self, action):
        """gym.Env step function."""
        obs, reward, done, info = self.env.step(action)
        return self._observation(obs), reward, done, info

""" print(source_env.observation_space)
print(source_env.reset())
print(f"SOURCE ENV: {source_env}")
 """
pixel_env = PixelObservationWrapper(source_env)
#print(f"PIXEL_ENV OBSERV SPACE: {pixel_env.observation_space}")
obs = pixel_env.reset()
""" print(f"OBS : {obs}")
print(f"OBS .SHAPE: {obs.shape}")
print(f"PIXEL ENV: {pixel_env}")
 """
#source_env = GrayScaleObservation(pixel_env)

#Works only if the parameter 'pixel_keys' was specified in the PixelObservationWrapper constructor
#obs_dict['state'].shape

# plt.imshow(obs_dict, cmap='gray', vmin=0, vmax=255) #grayscale colormap
# plt.show()

image_env = ImageToPyTorch(pixel_env)
image_env.observation_space

grayscale_env = Grayscale(pixel_env)
grayscale_env.observation_space
grayscale_env

resized_env = Resize(grayscale_env, 100, 100)
#print(resized_env.observation_space)

stackframes_env = StackFrames(resized_env, 4)
stackframes_env.observation_space

image_env = ImageToPyTorch(stackframes_env)
image_env.observation_space

pixel_target_env = PixelObservationWrapper(target_env)
pixel_target_env.observation_space

grayscale_target_env = Grayscale(pixel_target_env)
grayscale_target_env.observation_space
grayscale_target_env

resized_target_env = Resize(grayscale_target_env, 100, 100)
#print(resized_target_env.observation_space)

stackframes_target_env = StackFrames(resized_target_env, 4)
stackframes_target_env.observation_space

image_target_env = ImageToPyTorch(stackframes_target_env)
image_target_env.observation_space

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        print(f"N channels: {n_input_channels}")
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.Tanh(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.Tanh(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
)

# Save a checkpoint every x steps
checkpoint_callback = CheckpointCallback(
  save_freq=25_000,
  save_path="./logs/",
  name_prefix="trpo_model_lr_1e-2_150k",
  save_replay_buffer=True,
  save_vecnormalize=True,
)

""" source_env.set_udr_flag(True, 30) 
model = TRPO("MlpPolicy", image_env, policy_kwargs=policy_kwargs, verbose=1, batch_size=32, device="cpu", learning_rate=0.01)
trained_model = model.learn(total_timesteps=150_000, progress_bar=True, callback=checkpoint_callback)
source_env.set_udr_flag(False)
mean_reward, std = evaluate_policy(trained_model, Monitor(image_env), 50)
print(f"TRPO 150K timesteps with udr 30% mass var and lr=1-e2, s2s: mean reward={mean_reward}, std={std}")
mean_reward, std = evaluate_policy(trained_model, Monitor(image_target_env), 50)
print(f"TRPO 150K timesteps with udr 30% mass var and lr=1-e2, s2t: mean reward={mean_reward}, std={std}")
source_env.reset_masses_ranges() """

modelTRPO = TRPO.load("./logs/trpo_model_lr_1e-2_150k_25000_steps")
mean_reward, std_reward = evaluate_policy(modelTRPO, Monitor(image_env), 50)
print(f"\nMean reward on source environment : {mean_reward:.2f} +/- {std_reward:.2f}")
mean_reward, std_reward = evaluate_policy(modelTRPO, Monitor(image_target_env), 50)
print(f"\nMean reward on target environment : {mean_reward:.2f} +/- {std_reward:.2f}")

modelTRPO = TRPO.load("./logs/trpo_model_lr_1e-2_150k_50000_steps")
mean_reward, std_reward = evaluate_policy(modelTRPO, Monitor(image_env), 50)
print(f"\nMean reward on source environment : {mean_reward:.2f} +/- {std_reward:.2f}")
mean_reward, std_reward = evaluate_policy(modelTRPO, Monitor(image_target_env), 50)
print(f"\nMean reward on target environment : {mean_reward:.2f} +/- {std_reward:.2f}")

modelTRPO = TRPO.load("./logs/trpo_model_lr_1e-2_150k_75000_steps")
mean_reward, std_reward = evaluate_policy(modelTRPO, Monitor(image_env), 50)
print(f"\nMean reward on source environment : {mean_reward:.2f} +/- {std_reward:.2f}")
mean_reward, std_reward = evaluate_policy(modelTRPO, Monitor(image_target_env), 50)
print(f"\nMean reward on target environment : {mean_reward:.2f} +/- {std_reward:.2f}")

modelTRPO = TRPO.load("./logs/trpo_model_lr_1e-2_150k_100000_steps")
mean_reward, std_reward = evaluate_policy(modelTRPO, Monitor(image_env), 50)
print(f"\nMean reward on source environment : {mean_reward:.2f} +/- {std_reward:.2f}")
mean_reward, std_reward = evaluate_policy(modelTRPO, Monitor(image_target_env), 50)
print(f"\nMean reward on target environment : {mean_reward:.2f} +/- {std_reward:.2f}")

modelTRPO = TRPO.load("./logs/trpo_model_lr_1e-2_150k_125000_steps")
mean_reward, std_reward = evaluate_policy(modelTRPO, Monitor(image_env), 50)
print(f"\nMean reward on source environment : {mean_reward:.2f} +/- {std_reward:.2f}")
mean_reward, std_reward = evaluate_policy(modelTRPO, Monitor(image_target_env), 50)
print(f"\nMean reward on target environment : {mean_reward:.2f} +/- {std_reward:.2f}")
