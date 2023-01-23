import numpy as np
import gym

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