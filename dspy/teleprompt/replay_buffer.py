"""Replay buffer for TFLL RL."""

from collections import deque
import random
import numpy as np


class ReplayBuffer:
    """Experience replay buffer."""
    
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)
        self.total_added = 0
        
    def add(self, state, action, reward, logprob, adv):
        """Add experience."""
        exp = {
            'state': state,
            'action': action,
            'reward': reward,
            'logprob': logprob,
            'advantage': adv
        }
        self.buffer.append(exp)
        self.total_added += 1
    
    def sample_batch(self, batch_size=32):
        """Sample batch of experiences."""
        if len(self.buffer) == 0:
            return []
        size = min(batch_size, len(self.buffer))
        return random.sample(list(self.buffer), size)
    
    def add_trajectory(self, traj, gamma=0.99, lam=0.95):
        """Add entire trajectory."""
        advs = traj.compute_advantages(gamma, lam)
        for t in range(len(traj.actions)):
            self.add(
                traj.states[t],
                traj.actions[t],
                traj.rewards[t],
                traj.logprobs[t],
                advs[t] if t < len(advs) else 0.0
            )
    
    def __len__(self):
        return len(self.buffer)