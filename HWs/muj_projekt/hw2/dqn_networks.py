import torch
import torch.nn as nn


class DQNNetwork(nn.Module):
    """Fully-connected Q-network that outputs Q-values for our state.

    Input:  The state of our environment
    Output: Q-values for each possible action. We require fixed size output,
    so it will have to be for each action possible in the environment, which we
    will filter out through legal action mask.
    """

    def __init__(self, state_dim: int, n_actions: int, hidden_features=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, n_actions),
        )

    def forward(self, x: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        output_unmasked = self.net(x)
        #Give a value of -inf to illegal actions, to ensure they are not selected
        return torch.where(legal_mask, output_unmasked, torch.tensor(-1e9, device=x.device))

