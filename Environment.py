import gymnasium as gym
import numpy as np
from torch import dtype
import math

class GridWorld(gym.Env):
    def __init__(self, width: int = 30, height: int = 30, observation_size = 5, num_obstacles = 30, horizon=100, obstacles = None):
        self.width = width
        self.height = height
        self.horizon = horizon
        self.observation_size = observation_size

        self._agent_location = np.array([-1,-1], dtype=np.int32)

        if obstacles is not None:
            self._obstacles = obstacles
        else:
            self._obstacles = self.np_random.integers(0, self.width, size=(num_obstacles,2), dtype=np.int32)

        self._step_count = 0

        self._target = np.full((self.observation_size, self.observation_size), -1, dtype=np.int32)

        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(
                    low=0,  # 1 for out-of-bounds cells
                    high=1,  # 1 for obstacles, 0 for free space
                    shape=(5, 5),
                    dtype=np.float32
                ),
                "target": gym.spaces.Box(
                    low=0,  # 1 for out-of-bounds cells
                    high=1,  # 1 for obstacles, 0 for free space
                    shape=(5, 5),
                    dtype=np.float32
                )
            }
        )

        self.action_space = gym.spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([1, 0], dtype=np.int32),  # right
            1: np.array([0, 1], dtype=np.int32),  # up
            2: np.array([-1, 0], dtype=np.int32),  # left
            3: np.array([0, -1], dtype=np.int32),  # down
        }

    def _get_obs(self):
        obs_size = self.observation_size
        obs = np.full((obs_size, obs_size), 1, dtype=np.float32)  # Initialize with 1 for out-of-bounds
        for dx in range(-math.floor(obs_size/2), math.ceil(obs_size/2)):
            for dy in range(-math.floor(obs_size/2), math.ceil(obs_size/2)):
                grid_x = self._agent_location[0] + dx
                grid_y = self._agent_location[1] + dy
                if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
                    loc = np.array([grid_x, grid_y], dtype=np.float32)
                    if any(np.array_equal(loc, obstacle) for obstacle in self._obstacles):
                        obs[dx + 2, dy + 2] = 1  # Obstacle
                    else:
                        obs[dx + 2, dy + 2] = 0  # Free space
        return {"agent": obs, "target": self._target}

    def _get_info(self):
        return self._get_obs()

    def get_all_obs(self):
        prev_agent_loc = self._agent_location
        observations = []
        for x in range(self.width):
            for y in range(self.height):
                self._agent_location = np.array([x,y], dtype=np.int32)
                agent_on_obstacle = False
                for obstacle in self._obstacles:
                    if np.array_equal(obstacle,self._agent_location):
                        agent_on_obstacle = True
                        break
                if agent_on_obstacle:
                    continue
                observations.append(self._get_obs())

        self._agent_location = prev_agent_loc
        return observations

    def reset(self, target, seed: int | None = None):
        super().reset(seed=seed)
        self._step_count = 0

        # Choose the agent's location uniformly at random
        while True:
            self._agent_location = self.np_random.integers(0, self.width, size=2, dtype=np.int32)
            agent_on_obstacle = False
            for obs in self._obstacles:
                if np.array_equal(obs, self._agent_location):
                    agent_on_obstacle = True
            if not agent_on_obstacle:
                break

        self._target = target

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        direction = self._action_to_direction[action]

        prev_loc = self._agent_location
        self._agent_location = self._agent_location + direction.astype(np.int32)
        if self._agent_location[0] > self.width-1:
            self._agent_location[0] = np.int32(self.width-1)
        if self._agent_location[0] < 0:
            self._agent_location[0] = np.int32(0)
        if self._agent_location[1] > self.height-1:
            self._agent_location[1] = np.int32(self.height-1)
        if self._agent_location[1] < 0:
            self._agent_location[1] = np.int32(0)
        for obstacle in self._obstacles:
            if np.array_equal(obstacle,self._agent_location):
                self._agent_location = prev_loc
        self._agent_location = self._agent_location.astype(np.int32)

        self._step_count += 1

        if self._step_count >= self.horizon - 1:
            return self._get_obs(), 0, False, True, self._get_info()

        obs = self._get_obs()
        terminated = np.array_equal(obs["agent"], obs["target"])
        reward = 1 if terminated else 0  # the agent is only rewarded at the end of the episode
        return obs, reward, terminated, False, self._get_info()
