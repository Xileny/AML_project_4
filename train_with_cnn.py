from gym.wrappers.pixel_observation import PixelObservationWrapper
import gym
from env.custom_hopper import *

def main():
    """ source_env = PixelObservationWrapper(gym.make('CustomHopper-source-v0'))
    target_env = PixelObservationWrapper(gym.make('CustomHopper-target-v0')) """
    source_env = gym.make('CustomHopper-source-v0')
    target_env = gym.make('CustomHopper-target-v0')

    source_env = PixelObservationWrapper(source_env)

    """ policy = CustomCNN(env.action_space.n)
    model = PPO(policy, env, verbose=1) """

    """ obs = source_env.reset_model()
    print(obs) """

    source_env.close()
    target_env.close()


if __name__ == '__main__':
    main()