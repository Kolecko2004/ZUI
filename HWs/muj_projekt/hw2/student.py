from blockworld import BlockWorldEnv
import random
import time


class QLearning():
    def __init__(self, env: BlockWorldEnv):
        self.env = env
        self.Q = {}

    def _get_below(self, world):
        below = {}
        for stack in world.state:
            for i, block in enumerate(stack):
                below[int(block)] = int(stack[i + 1]) if i + \
                    1 < len(stack) else 0
        return below

    def _count_correct(self, current, goal):
        goal_below = self._get_below(goal)
        current_below = self._get_below(current)
        return sum(1 for b, v in goal_below.items() if current_below.get(b) == v)

    def train(self):
        alpha = 0.3
        gamma = 0.95
        epsilon_min = 0.02
        max_steps = 200
        n = self.env.n

        train_duration = 28.0
        start = time.time()

        while True:
            elapsed = time.time() - start
            if elapsed >= train_duration:
                break

            epsilon = max(epsilon_min,
                          1.0 - (1.0 - epsilon_min) * (elapsed / train_duration))

            s, _ = self.env.reset()

            for _ in range(max_steps):
                state_key = (s[0].get_state(), s[1].get_state())
                actions = s[0].get_actions()

                if random.random() < epsilon:
                    action = random.choice(actions)
                else:
                    best_val = float('-inf')
                    best_action = actions[0]
                    for a in actions:
                        val = self.Q.get((state_key, a), 0.0)
                        if val > best_val:
                            best_val = val
                            best_action = a
                    action = best_action

                s_, r, done, truncated, _ = self.env.step(action)
                next_key = (s_[0].get_state(), s_[1].get_state())

                phi_s = self._count_correct(s[0],  s[1]) / n
                phi_s_ = self._count_correct(s_[0], s_[1]) / n
                r_shaped = r + gamma * phi_s_ - phi_s

                current_q = self.Q.get((state_key, action), 0.0)

                if done:
                    td_target = r_shaped
                else:
                    next_actions = s_[0].get_actions()
                    max_next_q = max(self.Q.get((next_key, a), 0.0)
                                     for a in next_actions)
                    td_target = r_shaped + gamma * max_next_q

                self.Q[(state_key, action)] = (
                    current_q + alpha * (td_target - current_q)
                )

                s = s_
                if done:
                    break

    def act(self, s):
        state_key = (s[0].get_state(), s[1].get_state())
        actions = s[0].get_actions()

        best_val = float('-inf')
        best_action = actions[0]
        for a in actions:
            val = self.Q.get((state_key, a), 0.0)
            if val > best_val:
                best_val = val
                best_action = a
        return best_action
