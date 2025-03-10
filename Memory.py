from collections import namedtuple
from torch.nn import MSELoss
import numpy as np
import torch
import random

TransitionGCSL = namedtuple(
    'Transition',
    (
        'state',
        'latent_state',
	    'predictive_hidden_state',
        'next_latent_state',
	    'action',
	    'goal',
    )
)

Transition = namedtuple(
    'Transition',
    (
        'state',
        'latent_state',
	    'predictive_hidden_state',
	    'next_latent_state',
        'prediction_error'
    )
)

class ReplayMemory:
    def __init__(self, size):
        self.size = size
        self.buffer = []

    def sample(self, size = 1) -> list[TransitionGCSL]:
        sampled_indices = random.sample(range(len(self.buffer)), size)
        sampled_elements = [self.buffer[i] for i in sampled_indices]
        # Remove sampled elements
        self.buffer = [self.buffer[i] for i in range(len(self.buffer)) if i not in sampled_indices]
        return sampled_elements

    def append(self, state, latent_state, predictive_hidden_state, next_latent_state, action, goal):
        # only allow unique elements
        if any(
            np.array_equal(state, t.state) and
            np.array_equal(goal, t.goal) and
            np.array_equal(action, t.action)
            for t in self.buffer
        ):
            return 0

        if len(self.buffer) >= self.size:
            self.buffer.pop(0)  # Remove the oldest element
        self.buffer.append(TransitionGCSL(state, latent_state, predictive_hidden_state, next_latent_state, action, goal))
        return 1

    def insert_trajectory(self, trajectory, episode):
        # Relabel trajectory and insert into replay_memory
        # Schedule:
        for i, goal in enumerate(trajectory):
            if i > 5 and episode < 500:
                continue
            elif i > 10 and episode < 1000:
                continue
            elif i > 15 and episode < 1500:
                continue
            elif i > 20 and episode < 2000:
                continue
            for j in range(i):
                # Replace goal with "correct" subgoal
                transition = trajectory[j]
                if np.array_equal(transition.state, goal.state):
                    continue
                self.append(
                    transition.state,
					transition.latent_state,
					transition.predictive_hidden_state,
                    transition.next_latent_state,
					transition.action,
                    goal.latent_state
                )

    def __len__(self):
        return len(self.buffer)

class ExperienceMemory:
    def __init__(self, size):
        self.size = size
        self.buffer = []
        self.criterion = MSELoss()

    def sample(self) -> Transition:
        # return element with highest prediction error to be used as next goal
        max_error_idx = 0
        for idx, transition in enumerate(self.buffer):
            if transition.prediction_error > self.buffer[max_error_idx].prediction_error:
                max_error_idx = idx

        return self.buffer.pop(max_error_idx)

    def append(self, state, latent_state, predictive_hidden_state, next_latent_state, prediction_error=None):
        # only allow unique elements
        if any(
            np.array_equal(state, t.state) and
            np.array_equal(latent_state, t.latent_state) and
            np.array_equal(next_latent_state, t.next_latent_state)
            for t in self.buffer
        ):
            return 0

        if len(self.buffer) >= self.size:
            self.buffer.pop(0)  # Remove the oldest element
        if prediction_error is None:
            prediction_error = self.criterion(predictive_hidden_state, next_latent_state)
        self.buffer.append(Transition(state, latent_state, predictive_hidden_state, next_latent_state, prediction_error))
        return 1

    def __len__(self):
        return len(self.buffer)
