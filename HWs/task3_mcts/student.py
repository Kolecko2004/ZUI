import random
import time
import math
import numpy as np
import ox

class Node:
    def __init__(self, parent=None, action=None, untried_actions=None, player_who_made_move=None):
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = untried_actions if untried_actions is not None else []
        self.player_who_made_move = player_who_made_move

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param=1.414):
        choices_weights = []
        for c in self.children:
            exploit = c.value / c.visits
            explore = c_param * math.sqrt(math.log(self.visits) / c.visits)
            choices_weights.append(exploit + explore)
        return self.children[np.argmax(choices_weights)]

class MCTSBot:
    def __init__(self, play_as: int, time_limit: float):
        self.play_as = play_as
        self.time_limit = time_limit * 0.9

    def simulation(self, state):
        current_player = state.current_player()
        while not state.is_terminal():
            action = random.choice(list(state.get_actions()))
            state.apply_action(action)
        rewards = state.get_rewards()
        return rewards[current_player]

    def play_action(self, board):
        root = Node(
            untried_actions=list(board.get_actions()), 
            player_who_made_move=1 - board.current_player()
        )
        start_time = time.time()

        while (time.time() - start_time) < self.time_limit:
            node = root
            state = board.clone()

            while not state.is_terminal() and node.is_fully_expanded() and node.children:
                node = node.best_child()
                state.apply_action(node.action)

            if not state.is_terminal() and not node.is_fully_expanded():
                action = random.choice(node.untried_actions)
                node.untried_actions.remove(action)
                
                player_making_move = state.current_player()
                state.apply_action(action)
                
                child_node = Node(
                    parent=node, 
                    action=action, 
                    untried_actions=list(state.get_actions()),
                    player_who_made_move=player_making_move
                )
                node.children.append(child_node)
                node = child_node

            current_sim_player = state.current_player()
            simulation_value = self.simulation(state)
            
            rewards = [0, 0]
            rewards[current_sim_player] = simulation_value
            rewards[1 - current_sim_player] = -simulation_value

            while node is not None:
                node.visits += 1
                if node.parent is not None:
                    node.value += rewards[node.player_who_made_move]
                node = node.parent

        if not root.children:
            return random.choice(list(board.get_actions()))

        best_move_node = max(root.children, key=lambda c: c.visits)
        return best_move_node.action

if __name__ == '__main__':
    board = ox.Board(8)
    bots = [MCTSBot(0, 0.1), MCTSBot(1, 1.0)]

    while not board.is_terminal():
        current_player = board.current_player()
        current_player_mark = ox.MARKS_AS_CHAR[ox.PLAYER_TO_MARK[current_player]]

        current_bot = bots[current_player]
        a = current_bot.play_action(board)
        board.apply_action(a)

        print(f"{current_player_mark}: {a} -> \n{board}\n")