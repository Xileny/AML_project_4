"""Implementation of the Hopper environment supporting
domain randomization optimization."""
import csv
import pdb
from copy import deepcopy

import numpy as np
import gym
from gym import utils
from .mujoco_env import MujocoEnv

class CustomHopper(MujocoEnv, utils.EzPickle):
    

    def __init__(self, domain=None):
        MujocoEnv.__init__(self, 4)
        utils.EzPickle.__init__(self)

        self.udr = False
        self.percentage_mass_variability = 10
        self.masses_ranges = []

        self.original_masses = np.copy(self.sim.model.body_mass[1:])    # Default link masses

        if domain == 'source':  # Source environment has an imprecise torso mass (1kg shift)
            self.sim.model.body_mass[1] -= 1.0


    def set_random_parameters(self):
        """Set random masses
        TODO
        """
        self.set_parameters(self.sample_parameters())

    def sample_parameters(self):
        """Sample masses according to a domain randomization distribution
        TODO
        """
        ############## UNIFORM DISTRIBUTION ##############
        masses= [] 
        if(self.percentage_mass_variability > 99):
            self.percentage_mass_variability = 99

        if len(self.masses_ranges) == 0:
            i = 0
            for mass in self.original_masses[1:]:
                min_mass, max_mass = mass-(mass*self.percentage_mass_variability/100), mass+(mass*self.percentage_mass_variability/100)
                print(f"Range for mass {i+1}: [{min_mass:.3f}, {max_mass:.3f})")
                i += 1
                interval = [min_mass, max_mass]
                self.masses_ranges.append(interval)
                
        for interval in self.masses_ranges:
            sampled_mass = np.random.uniform(interval[0], interval[1])
            masses.append(sampled_mass)

        return masses

    def get_parameters(self):
        """Get value of mass for each link"""
        masses = np.array( self.sim.model.body_mass[2:] )
        return masses

    def set_parameters(self, task):
        """Set each hopper link's mass to a new value"""
        self.sim.model.body_mass[2:] = task

    def set_udr_flag(self, val=True, percentage_variability = 10):
        self.udr = val
        self.percentage_mass_variability = percentage_variability
    
    def reset_masses_ranges(self):
        self.masses_ranges = []

    def step(self, a):
        """Step the simulation to the next timestep

        Parameters
        ----------
        a : ndarray,
            action to be taken at the current timestep
        """
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()

        return ob, reward, done, {}

    def _get_obs(self):
        """Get current state"""
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat
        ])

    def reset_model(self):
        """Reset the environment to a random initial state"""

        if self.udr:
            #This method should be called after each episode
            self.set_random_parameters()

        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20



"""
    Registered environments
"""
gym.envs.register(
        id="CustomHopper-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
)

gym.envs.register(
        id="CustomHopper-source-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "source"}
)

gym.envs.register(
        id="CustomHopper-target-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "target"}
)

