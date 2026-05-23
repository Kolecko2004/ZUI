import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import numpy as np
from student import MCTSBot
import ox

def board_to_tensor(board):
    current_player = board.current_player()
    my_mark = ox.PLAYER_TO_MARK[current_player]
    opp_mark = ox.PLAYER_TO_MARK[1 - current_player]
    
    grid = np.zeros((board.size, board.size), dtype=np.float32)
    grid[board.board == my_mark] = 1.0
    grid[board.board == opp_mark] = -1.0
    
    return torch.tensor(grid).unsqueeze(0).unsqueeze(0)

class ValueNetwork(nn.Module):
    def __init__(self, board_size=8):
        super().__init__()
        # Zmenšení kanálů pro EXTRÉMNÍ zrychlení na CPU při evaluaci na Brute
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(32 * board_size * board_size, 128)
        self.dropout = nn.Dropout(0.3) # Zabrání memorování nazpaměť!
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x) # Aplikuje se jen při tréninku
        x = torch.tanh(self.fc2(x))
        return x

class NeuralMCTSBot(MCTSBot):
    def __init__(self, play_as: int, time_limit: float):
        super().__init__(play_as, time_limit)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = ValueNetwork().to(self.device)
        self.net.eval()
        
        model_path = "nn_mcts.pth"
        if os.path.exists(model_path):
            self.net.load_state_dict(torch.load(model_path, map_location=self.device))

    def simulation(self, board):
        if board.is_terminal():
            rewards = board.get_rewards()
            return rewards[board.current_player()]

        tensor_state = board_to_tensor(board).to(self.device)
        
        with torch.no_grad():
            value = self.net(tensor_state).item()
            
        return value