import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import ox
from nn_mcts import board_to_tensor, ValueNetwork
from student import MCTSBot
from nn_mcts import NeuralMCTSBot

def get_symmetries(tensor_state, value):
    """Z 1 stavu desky udělá 8 díky rotacím a zrcadlení."""
    symmetries = []
    # Převedeme zpět na 2D numpy pole pro snazší manipulaci
    grid = tensor_state.squeeze(0).numpy()

    for i in range(4):
        # Rotace 0, 90, 180, 270 stupňů
        rotated = np.rot90(grid, i)
        # Zrcadlové převrácení
        flipped = np.fliplr(rotated)
        
        symmetries.append((torch.tensor(rotated.copy()).unsqueeze(0), value))
        symmetries.append((torch.tensor(flipped.copy()).unsqueeze(0), value))
        
    return symmetries

def generate_games(num_games, time_limit=0.1): # Krátký čas pro masivní diverzitu her
    print(f"Generuji {num_games} rozmanitých her (čas na tah: {time_limit}s)...")
    training_data = []

    for i in range(num_games):
        board = ox.Board(8)
        bots = [MCTSBot(0, time_limit), MCTSBot(1, time_limit)]
        states = []
        players = []
        
        while not board.is_terminal():
            current = board.current_player()
            action = bots[current].play_action(board)
            states.append(board_to_tensor(board).squeeze(0))
            players.append(current)
            board.apply_action(action)
            
        rewards = board.get_rewards()
        
        for state_tensor, player in zip(states, players):
            value = torch.tensor([rewards[player]], dtype=torch.float32)
            sym_data = get_symmetries(state_tensor, value)
            training_data.extend(sym_data)
            
        if (i + 1) % 20 == 0:
            print(f"Odehráno {i + 1}/{num_games} her. Aktuální počet pozic: {len(training_data)}")
            
    return training_data

def train_network():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ValueNetwork().to(device)
    # L2 regularizace (weight_decay) zamezuje overfittingu
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4) 
    criterion = nn.MSELoss()

    # Zvýšeno na 500 her pro reálné pochopení strategie
    data = generate_games(num_games=500, time_limit=0.5) 

    print(f"Získáno {len(data)} pozic díky symetrii. Začínám trénink...")

    X = torch.stack([d[0] for d in data]).to(device)
    y = torch.stack([d[1] for d in data]).to(device)

    epochs = 15
    batch_size = 128 # Větší batch vyhlazuje učení
    dataset_size = len(X)

    net.train()
    for epoch in range(epochs):
        indices = torch.randperm(dataset_size)
        total_loss = 0
        
        for i in range(0, dataset_size, batch_size):
            batch_idx = indices[i:i + batch_size]
            batch_X = X[batch_idx]
            batch_y = y[batch_idx]
            
            optimizer.zero_grad()
            outputs = net(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / (dataset_size/batch_size):.4f}")
        
    torch.save(net.state_dict(), "nn_mcts.pth")
    print("Trénink dokončen. Odevzdej tento nový 'nn_mcts.pth'.")

if __name__ == "__main__":
    train_network()