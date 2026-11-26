import numpy as np


class HistoryBuffer:

    def __init__(self, history_length, obs_dim):
        self.history_length = history_length
        self.obs_dim = obs_dim

        self.buffer = np.zeros((history_length, obs_dim), dtype=np.float32)
        self.initialized = False

    def reset(self):
        self.buffer.fill(0.0)
        self.initialized = False

    def append(self, obs):
        obs = np.asarray(obs, dtype=np.float32)

        if obs.shape != (self.obs_dim,):
            raise ValueError(f"Expected observation shape {(self.obs_dim,)}, got {obs.shape}")

        if not self.initialized:
            self.buffer[:] = obs
            self.initialized = True
        else:
            self.buffer[:-1] = self.buffer[1:].copy()
            self.buffer[-1] = obs

    def get(self):
        return self.buffer.flatten()
