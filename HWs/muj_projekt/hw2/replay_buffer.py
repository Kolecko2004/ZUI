import numpy as np
from collections import deque
from typing import NamedTuple, List

import torch


class Transition(NamedTuple):
    state: torch.Tensor
    mask: torch.Tensor
    action: torch.Tensor
    next_state: torch.Tensor
    next_mask: torch.Tensor
    reward: torch.Tensor


class ReplayBuffer:
    """Cyclic fixed-size buffer for storing and sampling experience transitions.
    We maintain two buffers. Since our task has sparse rewards, we maintain
    terminal transitions, which are highly important for our learning process, in
    a separate buffer."""

    def __init__(self, capacity: int, terminal_ratio: float):
        self.terminal_ratio = terminal_ratio
        self.buffer: deque[Transition] = deque(maxlen=capacity)
        self.terminal_buffer: deque[Transition] = deque(maxlen=int(terminal_ratio * capacity))

    def push(self, state: torch.Tensor, mask: torch.Tensor, action: int,
             next_state: torch.Tensor, next_mask: torch.Tensor,  reward: float) -> None:
        """Store a single transition. Overwrites the oldest entry when full."""
        #If next state is None, it is a terminal transition
        if next_state is None:
            self.terminal_buffer.append(Transition(state, mask, action, next_state, next_mask, reward))
        else:
            self.buffer.append(Transition(state, mask, action, next_state, next_mask, reward))

    def sample(self, n: int, rng: np.random.Generator) -> List[Transition]:
        """Return n transitions sampled uniformly at random (without replacement)."""
        #The logic is as follows. We first attempt to sample int(n * terminal_ratio)
        # samples from the terminal buffer. The remaining size is then filled
        # with samples from the regular buffer.
        terminal_to_sample = min(len(self.terminal_buffer), int(n * self.terminal_ratio))
        rest = n - terminal_to_sample
        #If the regular buffer does not have enough samples to fill, add some more
        # samples from the terminal buffer
        if len(self.buffer) < rest:
            terminal_to_sample += rest - len(self.buffer)
            rest = len(self.buffer)
        #If after this additon the terminal buffer does not hold enough elements
        # we cannot produce n elements. Raise an error:
        if len(self.terminal_buffer) < terminal_to_sample:
            raise ValueError(f"The buffer cannot provide the requested samples. Requested was (ordered as terminal, regular): {terminal_to_sample}, {len(self.buffer)}. The buffer holds: {len(self.terminal_buffer)}, {len(self.buffer)}")
        #Workaroud, since numpy rng would try (and fail) to convert the buffer to a np.ndarray
        terminal_buffer_indices = rng.choice(len(self.terminal_buffer), size=terminal_to_sample, replace=False)
        buffer_indices = rng.choice(len(self.buffer), size=rest, replace=False)
        samples = [self.terminal_buffer[i] for i in terminal_buffer_indices] + [self.buffer[i] for i in buffer_indices]
        return samples

    def __len__(self) -> int:
        return len(self.buffer) + len(self.terminal_buffer)

    def getstate(self):
        return {'terminal_buffer': self.terminal_buffer, 'buffer': self.buffer}

    def setstate(self, state):
        self.buffer = state['buffer']
        self.terminal_buffer = state['terminal_buffer']
