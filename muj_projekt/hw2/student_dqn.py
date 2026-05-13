import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import json
import random
import torch

from dataclasses import dataclass
from argparse import ArgumentParser
from itertools import count
from typing import Iterable, Tuple

from blockworld import BlockWorldEnv
from dqn_networks import DQNNetwork
from replay_buffer import ReplayBuffer, Transition

# ---------------------------------------------------------------------------
# CLI arguments – do not modify
# ---------------------------------------------------------------------------
parser = ArgumentParser()
parser.add_argument("--num_blocks",         type=int,  default=5)
parser.add_argument("--model_restore_path", type=str,  default="checkpoints")
parser.add_argument("--restore_episode",    type=int,  default=-1)
parser.add_argument("--no_train",           action="store_true")
parser.add_argument("--test_problems",      type=int,  default=1000)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class DQNConfig:
    # Do not delete these!
    seed: int = 42
    batch_size: int = 128
    hidden_features: int = 256
    target_net_update: float = 0.005
    lr: float = 3e-4
    replay_capacity: int = 10000
    replay_terminal_ratio: float = 0.1
    num_episodes: int = 5000
    save_interval: int = 500
    # Additional parameters
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.999   # multiplicative per episode
    grad_clip: float = 10.0

# ---------------------------------------------------------------------------
# DQN agent
# ---------------------------------------------------------------------------
class DQN():
    # don't modify the methods' signatures!
    def __init__(self, env: BlockWorldEnv, config: DQNConfig, N: int):
        self.num_blocks = N
        self.env        = env
        self.config     = config
        self.device     = torch.device('cpu')

        self.steps_done = 0
        self.episodes   = 0

        self.memory = ReplayBuffer(config.replay_capacity,
                                   config.replay_terminal_ratio)

        # Fix seeds for reproducibility
        self.rng = np.random.default_rng(config.seed)
        torch.manual_seed(config.seed)
        random.seed(config.seed)

        # Fixed-size state vector: 2 * N * (N+1)
        self.state_dim = 2 * N * (N + 1)
        # N actions for each of the N blocks (move to any other block or ground)
        self.n_actions = N * N

        self.q_net      = DQNNetwork(self.state_dim, self.n_actions,
                                     config.hidden_features)
        self.target_net = DQNNetwork(self.state_dim, self.n_actions,
                                     config.hidden_features)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), config.lr)

    def _get_below_map(self, world) -> dict:
        """Map each block (int) to what is directly below it (0 = ground)."""
        below = {}
        for stack in world.state:
            for i, block in enumerate(stack):
                below[int(block)] = int(stack[i + 1]) if i + 1 < len(stack) else 0
        return below

    def _encode_below(self, below_map: dict) -> torch.Tensor:
        """One-hot encode the below-map into a flat vector of size N*(N+1)."""
        N   = self.num_blocks
        vec = torch.zeros(N * (N + 1))
        for b in range(1, N + 1):
            val = below_map.get(b, 0)          # 0..N
            vec[(b - 1) * (N + 1) + val] = 1.0
        return vec

    def _action_to_network_index(self, action: Tuple[int, int]) -> int:
        """Inverse of network_action_to_action."""
        what, where = action
        col = where - int(where > what)        # skip the 'itself' slot
        return (what - 1) * self.num_blocks + col

    def _count_correct(self, world, goal) -> int:
        """Count blocks whose 'below' relation already matches the goal."""
        goal_below  = self._get_below_map(goal)
        world_below = self._get_below_map(world)
        return sum(1 for b, v in goal_below.items() if world_below.get(b) == v)

    def state_to_tensor(self, state) -> torch.Tensor:
        """Convert (current_world, goal_world) into a flat tensor."""
        current_world, goal_world = state
        cur_enc  = self._encode_below(self._get_below_map(current_world))
        goal_enc = self._encode_below(self._get_below_map(goal_world))
        return torch.cat([cur_enc, goal_enc])

    def create_legal_mask(self, actions: Iterable) -> torch.Tensor:
        """Binary mask – True for every (what, where) pair that is currently legal."""
        all_actions = np.zeros(self.n_actions)
        for (what, where) in actions:
            row_offset = (what - 1) * self.num_blocks
            col_offset = where - int(where > what)
            all_actions[row_offset + col_offset] = 1
        return torch.tensor(all_actions, dtype=torch.bool)

    def compute_reward_shaping(self, state, next_state) -> float:
        """Potential-based reward shaping: F(s,s') = gamma * Phi(s') - Phi(s)"""
        current_world, goal_world = state
        next_world,    _          = next_state
        N = self.num_blocks

        phi_s  = self._count_correct(current_world, goal_world) / N
        phi_s_ = self._count_correct(next_world,    goal_world) / N

        return self.config.gamma * phi_s_ - phi_s

    def network_action_to_action(self, network_action: np.ndarray) -> Tuple[int, int]:
        """Convert action index returned by the network Q-values back to (what, where)."""
        network_action = int(network_action)
        what  = (network_action // self.num_blocks) + 1
        where =  network_action  % self.num_blocks
        where += (where >= what)
        return (what, where)

    def _optimize(self):
        """Sample a mini-batch from the replay buffer and do one gradient step."""
        try:
            transitions = self.memory.sample(self.config.batch_size, self.rng)
        except ValueError:
            return   # buffer not warm enough yet

        batch = Transition(*zip(*transitions))

        non_terminal_mask = torch.tensor(
            [s is not None for s in batch.next_state],
            dtype=torch.bool, device=self.device
        )

        state_batch  = torch.cat(batch.state)
        mask_batch   = torch.cat(batch.mask)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        self.q_net.train()
        q_values = self.q_net(state_batch, mask_batch)
        q_sa     = q_values.gather(1, action_batch).squeeze(1)

        next_q = torch.zeros(self.config.batch_size, device=self.device)
        if non_terminal_mask.any():
            nt_states = torch.cat([s for s in batch.next_state if s is not None])
            nt_masks  = torch.cat([m for m in batch.next_mask  if m is not None])
            with torch.no_grad():
                next_q[non_terminal_mask] = (
                    self.target_net(nt_states, nt_masks).max(1).values
                )

        td_target = reward_batch + self.config.gamma * next_q
        loss = self.criterion(q_sa, td_target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(),
                                       self.config.grad_clip)
        self.optimizer.step()

        tau = self.config.target_net_update
        for qp, tp in zip(self.q_net.parameters(),
                          self.target_net.parameters()):
            tp.data.copy_(tau * qp.data + (1.0 - tau) * tp.data)

    def train(self):
        epsilon = self.config.epsilon_start

        for i_episode in range(self.config.num_episodes):
            raw_state, _ = self.env.reset()
            state = self.state_to_tensor(raw_state).unsqueeze(0)
            mask  = self.create_legal_mask(raw_state[0].get_actions()).unsqueeze(0)

            for t in count():
                if self.rng.random() < epsilon:
                    action = random.choice(raw_state[0].get_actions())
                else:
                    self.q_net.eval()
                    with torch.no_grad():
                        q_values      = self.q_net(state, mask)
                        network_action = q_values.argmax(dim=1).item()
                    action = self.network_action_to_action(network_action)

                raw_next_state, reward, terminated, truncated, _ = (
                    self.env.step(action)
                )

                reward_t   = torch.tensor([reward], device=self.device, dtype=torch.float)
                reward_t  += self.compute_reward_shaping(raw_state, raw_next_state)

                if terminated:
                    next_state = None
                    next_mask  = None
                else:
                    next_state = self.state_to_tensor(raw_next_state).unsqueeze(0)
                    next_mask  = self.create_legal_mask(
                        raw_next_state[0].get_actions()
                    ).unsqueeze(0)

                net_idx       = self._action_to_network_index(action)
                action_tensor = torch.tensor([[net_idx]], device=self.device,
                                             dtype=torch.long)

                self.memory.push(state, mask, action_tensor,
                                 next_state, next_mask, reward_t)

                self._optimize()

                state     = next_state
                raw_state = raw_next_state
                mask      = next_mask

                if terminated:
                    self.episodes += 1
                    if self.episodes % 100 == 0:
                        print(f"Episode {self.episodes:5d} | "
                              f"steps={t + 1:4d} | eps={epsilon:.4f}")
                    if (self.config.save_interval > 0
                            and self.episodes % self.config.save_interval == 0):
                        save_model(self, self.episodes)
                    break

            epsilon = max(self.config.epsilon_end,
                          epsilon * self.config.epsilon_decay)

        print('Complete')

    def act(self, state):
        """Greedy policy used during evaluation (no exploration)."""
        self.q_net.eval()
        with torch.no_grad():
            state_tensor = self.state_to_tensor(state).unsqueeze(0)
            mask         = self.create_legal_mask(
                state[0].get_actions()
            ).unsqueeze(0)
            q_values      = self.q_net(state_tensor, mask)
            network_action = q_values.argmax(dim=1).item()
        return self.network_action_to_action(network_action)

# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------
def load_model(model_dir: str, N: int) -> DQN:
    with open(os.path.join(model_dir, "config.json"), 'r') as f:
        config_dict = json.load(f)
    config = DQNConfig(**config_dict)

    env   = BlockWorldEnv(N)
    model = DQN(env, config, N)

    model.q_net.load_state_dict(
        torch.load(os.path.join(model_dir, "q_net.pt"),
                   weights_only=True, map_location=model.device)
    )
    model.target_net.load_state_dict(
        torch.load(os.path.join(model_dir, "target_net.pt"),
                   weights_only=True, map_location=model.device)
    )
    print(f"Loaded weights from '{model_dir}/'")
    return model

def save_model(model: DQN, episode: int, save_dir: str = "checkpoints") -> None:
    os.makedirs(save_dir, exist_ok=True)
    config_path = os.path.join(save_dir, "config.json")
    env_path    = os.path.join(save_dir, "env_spec.pt")
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            json.dump(vars(model.config), f, indent=2)
    if not os.path.exists(env_path):
        torch.save(model.num_blocks, env_path)
    torch.save(
        (model.rng.bit_generator.state, torch.get_rng_state()),
        os.path.join(save_dir, f"rng_states_ep{episode}.pt")
    )
    torch.save(model.q_net.state_dict(),
               os.path.join(save_dir, f"q_net_ep{episode}.pt"))
    torch.save(model.target_net.state_dict(),
               os.path.join(save_dir, f"target_net_ep{episode}.pt"))
    print(f"Saved weights at episode {episode} to '{save_dir}/'")

def nn_learner_setup(N: int, model_dir: str = None) -> DQN:
    env = BlockWorldEnv(N)
    if model_dir:
        return load_model(model_dir, N)
    config = DQNConfig()
    return DQN(env, config, N)

if __name__ == '__main__':
    args = parser.parse_args()

    if args.restore_episode > 0 and args.model_restore_path:
        qlearning = nn_learner_setup(args.num_blocks, args.model_restore_path)
    else:
        qlearning = nn_learner_setup(args.num_blocks)

    if not args.no_train:
        qlearning.train()

    test_env      = BlockWorldEnv(args.num_blocks)
    test_problems = args.test_problems
    solved        = 0
    avg_steps     = []

    for test_id in range(test_problems):
        s, _ = test_env.reset()
        done  = False

        for step in range(50):
            a = qlearning.act(s)
            s_, r, done, truncated, _ = test_env.step(a)
            s = s_
            if done:
                solved += 1
                avg_steps.append(step + 1)
                break

    avg_steps = sum(avg_steps) / len(avg_steps) if avg_steps else float('inf')
    print(f"\nSolved {solved}/{test_problems} problems, "
          f"with average number of steps {avg_steps}.")

# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["OMP_NUM_THREADS"] = "1"

# import numpy as np
# import json
# import random
# import torch

# from dataclasses import dataclass
# from argparse import ArgumentParser
# from itertools import count
# from typing import Iterable, Tuple

# from muj_projekt.hw2.blockworld import BlockWorldEnv
# from muj_projekt.hw2.dqn_networks import DQNNetwork
# from muj_projekt.hw2.replay_buffer import ReplayBuffer, Transition


# # ---------------------------------------------------------------------------
# # CLI arguments – do not modify
# # ---------------------------------------------------------------------------
# parser = ArgumentParser()
# parser.add_argument("--num_blocks",         type=int,  default=5)
# parser.add_argument("--model_restore_path", type=str,  default="checkpoints")
# parser.add_argument("--restore_episode",    type=int,  default=-1)
# parser.add_argument("--no_train",           action="store_true")
# parser.add_argument("--test_problems",      type=int,  default=1000)


# # ---------------------------------------------------------------------------
# # Configuration
# # ---------------------------------------------------------------------------
# @dataclass
# class DQNConfig:
#     # Do not delete these!
#     seed: int = 42
#     batch_size: int = 128
#     hidden_features: int = 256
#     target_net_update: float = 0.005
#     lr: float = 3e-4
#     replay_capacity: int = 10000
#     replay_terminal_ratio: float = 0.1
#     num_episodes: int = 5000
#     save_interval: int = 500
#     # Additional parameters
#     gamma: float = 0.99
#     epsilon_start: float = 1.0
#     epsilon_end: float = 0.05
#     epsilon_decay: float = 0.999   # multiplicative per episode
#     grad_clip: float = 10.0


# # ---------------------------------------------------------------------------
# # DQN agent
# # ---------------------------------------------------------------------------
# class DQN():
#     # don't modify the methods' signatures!

#     def __init__(self, env: BlockWorldEnv, config: DQNConfig, N: int):
#         """
#         State encoding:
#           For each block b in 1..N we store what sits directly below b
#           (0 = ground, 1..N = another block).  We one-hot encode this value
#           (N+1 possible values) and concatenate for all blocks.
#           We do this for the CURRENT state AND for the GOAL state.

#           state_dim = 2 * N * (N + 1)
#         """
#         self.num_blocks = N
#         self.env        = env
#         self.config     = config
#         self.device     = torch.device('cpu')

#         self.steps_done = 0
#         self.episodes   = 0

#         self.memory = ReplayBuffer(config.replay_capacity,
#                                    config.replay_terminal_ratio)

#         # Fix seeds for reproducibility
#         self.rng = np.random.default_rng(config.seed)
#         torch.manual_seed(config.seed)
#         random.seed(config.seed)

#         # Fixed-size state vector: 2 * N * (N+1)
#         self.state_dim = 2 * N * (N + 1)
#         # N actions for each of the N blocks (move to any other block or ground)
#         self.n_actions = N * N

#         self.q_net      = DQNNetwork(self.state_dim, self.n_actions,
#                                      config.hidden_features)
#         self.target_net = DQNNetwork(self.state_dim, self.n_actions,
#                                      config.hidden_features)
#         self.target_net.load_state_dict(self.q_net.state_dict())
#         self.target_net.eval()

#         self.criterion = torch.nn.MSELoss()
#         self.optimizer = torch.optim.Adam(self.q_net.parameters(), config.lr)

#     # ------------------------------------------------------------------ helpers

#     def _get_below_map(self, world) -> dict:
#         """Map each block (int) to what is directly below it (0 = ground)."""
#         below = {}
#         for stack in world.state:
#             for i, block in enumerate(stack):
#                 below[int(block)] = int(stack[i + 1]) if i + 1 < len(stack) else 0
#         return below

#     def _encode_below(self, below_map: dict) -> torch.Tensor:
#         """
#         One-hot encode the below-map into a flat vector of size N*(N+1).
#         For block b, positions [(b-1)*(N+1) .. b*(N+1)-1] are the one-hot slot.
#         Index 0 within the slot = on the ground; index k = block k below it.
#         """
#         N   = self.num_blocks
#         vec = torch.zeros(N * (N + 1))
#         for b in range(1, N + 1):
#             val = below_map.get(b, 0)          # 0..N
#             vec[(b - 1) * (N + 1) + val] = 1.0
#         return vec

#     def _action_to_network_index(self, action: Tuple[int, int]) -> int:
#         """Inverse of network_action_to_action."""
#         what, where = action
#         col = where - int(where > what)        # skip the 'itself' slot
#         return (what - 1) * self.num_blocks + col

#     def _count_correct(self, world, goal) -> int:
#         """Count blocks whose 'below' relation already matches the goal."""
#         goal_below  = self._get_below_map(goal)
#         world_below = self._get_below_map(world)
#         return sum(1 for b, v in goal_below.items() if world_below.get(b) == v)

#     # ------------------------------------------------------------------ public API

#     def state_to_tensor(self, state) -> torch.Tensor:
#         """
#         Convert (current_world, goal_world) into a flat tensor.

#         Encoding: for each block, store a one-hot vector of 'what is below it'
#         (N+1 options: ground or one of the N blocks).
#         Concatenate current encoding + goal encoding → shape (state_dim,).
#         """
#         current_world, goal_world = state
#         cur_enc  = self._encode_below(self._get_below_map(current_world))
#         goal_enc = self._encode_below(self._get_below_map(goal_world))
#         return torch.cat([cur_enc, goal_enc])

#     def create_legal_mask(self, actions: Iterable) -> torch.Tensor:
#         """Binary mask – True for every (what, where) pair that is currently legal."""
#         all_actions = np.zeros(self.n_actions)
#         for (what, where) in actions:
#             row_offset = (what - 1) * self.num_blocks
#             col_offset = where - int(where > what)
#             all_actions[row_offset + col_offset] = 1
#         return torch.tensor(all_actions, dtype=torch.bool)

#     def compute_reward_shaping(self, state, next_state) -> float:
#         """
#         Potential-based reward shaping:  F(s,s') = gamma * Phi(s') - Phi(s)

#         Phi(s) = (# blocks whose 'below' relation matches the goal) / N

#         This is guaranteed not to change the optimal policy (Ng et al., 1999)
#         while providing a dense signal that guides learning toward the goal.
#         """
#         current_world, goal_world = state
#         next_world,    _          = next_state
#         N = self.num_blocks

#         phi_s  = self._count_correct(current_world, goal_world) / N
#         phi_s_ = self._count_correct(next_world,    goal_world) / N

#         return self.config.gamma * phi_s_ - phi_s

#     def network_action_to_action(self, network_action: np.ndarray) -> Tuple[int, int]:
#         """Convert action index returned by the network Q-values back to (what, where)."""
#         network_action = int(network_action)
#         what  = (network_action // self.num_blocks) + 1
#         where =  network_action  % self.num_blocks
#         where += (where >= what)
#         return (what, where)

#     # ------------------------------------------------------------------ training

#     def _optimize(self):
#         """Sample a mini-batch from the replay buffer and do one gradient step."""
#         try:
#             transitions = self.memory.sample(self.config.batch_size, self.rng)
#         except ValueError:
#             return   # buffer not warm enough yet

#         batch = Transition(*zip(*transitions))

#         non_terminal_mask = torch.tensor(
#             [s is not None for s in batch.next_state],
#             dtype=torch.bool, device=self.device
#         )

#         state_batch  = torch.cat(batch.state)           # (B, state_dim)
#         mask_batch   = torch.cat(batch.mask)            # (B, n_actions)
#         action_batch = torch.cat(batch.action)          # (B, 1)
#         reward_batch = torch.cat(batch.reward)          # (B,)

#         # Q(s, a)  – only for the chosen action
#         self.q_net.train()
#         q_values = self.q_net(state_batch, mask_batch)  # (B, n_actions)
#         q_sa     = q_values.gather(1, action_batch).squeeze(1)  # (B,)

#         # max_a' Q_target(s', a')  for non-terminal transitions
#         next_q = torch.zeros(self.config.batch_size, device=self.device)
#         if non_terminal_mask.any():
#             nt_states = torch.cat([s for s in batch.next_state if s is not None])
#             nt_masks  = torch.cat([m for m in batch.next_mask  if m is not None])
#             with torch.no_grad():
#                 next_q[non_terminal_mask] = (
#                     self.target_net(nt_states, nt_masks).max(1).values
#                 )

#         td_target = reward_batch + self.config.gamma * next_q

#         loss = self.criterion(q_sa, td_target.detach())

#         self.optimizer.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(self.q_net.parameters(),
#                                        self.config.grad_clip)
#         self.optimizer.step()

#         # Soft update target network
#         tau = self.config.target_net_update
#         for qp, tp in zip(self.q_net.parameters(),
#                           self.target_net.parameters()):
#             tp.data.copy_(tau * qp.data + (1.0 - tau) * tp.data)

#     def train(self):
#         epsilon = self.config.epsilon_start

#         for i_episode in range(self.config.num_episodes):
#             raw_state, _ = self.env.reset()
#             state = self.state_to_tensor(raw_state).unsqueeze(0)
#             mask  = self.create_legal_mask(raw_state[0].get_actions()).unsqueeze(0)

#             for t in count():
#                 # ---- action selection (epsilon-greedy) --------------------
#                 if self.rng.random() < epsilon:
#                     action = random.choice(raw_state[0].get_actions())
#                 else:
#                     self.q_net.eval()
#                     with torch.no_grad():
#                         q_values      = self.q_net(state, mask)
#                         network_action = q_values.argmax(dim=1).item()
#                     action = self.network_action_to_action(network_action)

#                 # ---- environment step ------------------------------------
#                 raw_next_state, reward, terminated, truncated, _ = (
#                     self.env.step(action)
#                 )

#                 reward_t   = torch.tensor([reward], device=self.device,
#                                           dtype=torch.float)
#                 reward_t  += self.compute_reward_shaping(raw_state, raw_next_state)

#                 if terminated:
#                     next_state = None
#                     next_mask  = None
#                 else:
#                     next_state = self.state_to_tensor(raw_next_state).unsqueeze(0)
#                     next_mask  = self.create_legal_mask(
#                         raw_next_state[0].get_actions()
#                     ).unsqueeze(0)

#                 net_idx       = self._action_to_network_index(action)
#                 action_tensor = torch.tensor([[net_idx]], device=self.device,
#                                              dtype=torch.long)

#                 self.memory.push(state, mask, action_tensor,
#                                  next_state, next_mask, reward_t)

#                 # ---- optimisation ----------------------------------------
#                 self._optimize()

#                 state     = next_state
#                 raw_state = raw_next_state
#                 mask      = next_mask

#                 if terminated:
#                     self.episodes += 1
#                     if self.episodes % 100 == 0:
#                         print(f"Episode {self.episodes:5d} | "
#                               f"steps={t + 1:4d} | eps={epsilon:.4f}")
#                     if (self.config.save_interval > 0
#                             and self.episodes % self.config.save_interval == 0):
#                         save_model(self, self.episodes)
#                     break

#             # epsilon decay (multiplicative, floored at epsilon_end)
#             epsilon = max(self.config.epsilon_end,
#                           epsilon * self.config.epsilon_decay)

#         print('Complete')

#     def act(self, state):
#         """Greedy policy used during evaluation (no exploration)."""
#         self.q_net.eval()
#         with torch.no_grad():
#             state_tensor = self.state_to_tensor(state).unsqueeze(0)
#             mask         = self.create_legal_mask(
#                 state[0].get_actions()
#             ).unsqueeze(0)
#             q_values      = self.q_net(state_tensor, mask)
#             network_action = q_values.argmax(dim=1).item()
#         return self.network_action_to_action(network_action)


# # ---------------------------------------------------------------------------
# # Persistence helpers (do not modify signatures)
# # ---------------------------------------------------------------------------

# def load_model(model_dir: str, N: int) -> DQN:
#     """Restore a model from a directory containing config.json, q_net.pt, target_net.pt."""
#     with open(os.path.join(model_dir, "config.json"), 'r') as f:
#         config_dict = json.load(f)
#     config = DQNConfig(**config_dict)

#     env   = BlockWorldEnv(N)
#     model = DQN(env, config, N)

#     model.q_net.load_state_dict(
#         torch.load(os.path.join(model_dir, "q_net.pt"),
#                    weights_only=True, map_location=model.device)
#     )
#     model.target_net.load_state_dict(
#         torch.load(os.path.join(model_dir, "target_net.pt"),
#                    weights_only=True, map_location=model.device)
#     )
#     print(f"Loaded weights from '{model_dir}/'")
#     return model


# def save_model(model: DQN, episode: int, save_dir: str = "checkpoints") -> None:
#     """Save policy and target network weights to disk."""
#     os.makedirs(save_dir, exist_ok=True)
#     config_path = os.path.join(save_dir, "config.json")
#     env_path    = os.path.join(save_dir, "env_spec.pt")
#     if not os.path.exists(config_path):
#         with open(config_path, 'w') as f:
#             json.dump(vars(model.config), f, indent=2)
#     if not os.path.exists(env_path):
#         torch.save(model.num_blocks, env_path)
#     torch.save(
#         (model.rng.bit_generator.state, torch.get_rng_state()),
#         os.path.join(save_dir, f"rng_states_ep{episode}.pt")
#     )
#     torch.save(model.q_net.state_dict(),
#                os.path.join(save_dir, f"q_net_ep{episode}.pt"))
#     torch.save(model.target_net.state_dict(),
#                os.path.join(save_dir, f"target_net_ep{episode}.pt"))
#     print(f"Saved weights at episode {episode} to '{save_dir}/'")


# def nn_learner_setup(N: int, model_dir: str = None) -> DQN:
#     env = BlockWorldEnv(N)
#     if model_dir:
#         return load_model(model_dir, N)
#     config = DQNConfig()
#     return DQN(env, config, N)


# # ---------------------------------------------------------------------------
# # Entry point
# # ---------------------------------------------------------------------------
# if __name__ == '__main__':
#     args = parser.parse_args()

#     if args.restore_episode > 0 and args.model_restore_path:
#         qlearning = nn_learner_setup(args.num_blocks, args.model_restore_path)
#     else:
#         qlearning = nn_learner_setup(args.num_blocks)

#     if not args.no_train:
#         qlearning.train()

#     # ---- Evaluation ---------------------------------------------------------
#     test_env      = BlockWorldEnv(args.num_blocks)
#     test_problems = args.test_problems
#     solved        = 0
#     avg_steps     = []

#     for test_id in range(test_problems):
#         s, _ = test_env.reset()
#         done  = False

#         print(f"\nProblem {test_id}:")
#         print(f"{s[0]} -> {s[1]}")

#         for step in range(50):
#             a = qlearning.act(s)
#             print(f"{a}: {s[0]}")
#             s_, r, done, truncated, _ = test_env.step(a)
#             s = s_
#             if done:
#                 solved += 1
#                 avg_steps.append(step + 1)
#                 break

#     avg_steps = sum(avg_steps) / len(avg_steps) if avg_steps else float('inf')
#     print(f"\nSolved {solved}/{test_problems} problems, "
#           f"with average number of steps {avg_steps}.")