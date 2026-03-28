from blockworld import BlockWorld
from queue import PriorityQueue

class BlockWorldHeuristic(BlockWorld):
    def __init__(self, num_blocks=5, state=None):
        BlockWorld.__init__(self, num_blocks, state)

    def heuristic(self, goal):
        self_state = self.get_state()
        goal_state = goal.get_state()

        goal_support = {}
        for stack in goal_state:
            for i in range(len(stack)):
                block = stack[i]
                under = stack[i + 1] if i + 1 < len(stack) else None
                goal_support[block] = under

        misplaced = 0
        
        for stack in self_state:
            stack_correct = True
            for i in range(len(stack) - 1, -1, -1):
                block = stack[i]
                under = stack[i + 1] if i + 1 < len(stack) else None

                if stack_correct:
                    if goal_support.get(block) == under:
                        pass
                    else:
                        stack_correct = False
                        misplaced += 1
                else:
                    misplaced += 1

        return misplaced
        
class AStar():
    def search(self, start, goal):
        counter = 0
        queue = PriorityQueue()

        start_h = start.heuristic(goal)
        queue.put((start_h, counter, start, []))

        def get_state_key(state):
            return tuple(sorted(tuple(stack) for stack in state if stack))

        best_g = {get_state_key(start.get_state()): 0}
        goal_key = get_state_key(goal.get_state())

        while not queue.empty():

            f, _, current, path = queue.get()
            current_state_key = get_state_key(current.get_state())
            g = len(path)

            if best_g.get(current_state_key, float('inf')) < g:
                continue

            if current_state_key == goal_key:
                return path

            for action, neighbor in current.get_neighbors():
                new_g = g + 1
                n_state_key = get_state_key(neighbor.get_state())

                if new_g < best_g.get(n_state_key, float('inf')):
                    best_g[n_state_key] = new_g
                    new_h = neighbor.heuristic(goal)
                    new_f = new_g + new_h
                    counter += 1
                    queue.put((new_f, counter, neighbor, path + [action]))

        return None
    
def format_state(state):
    return [[int(b) for b in stack] for stack in state]

def format_action(action):
    return (int(action[0]), int(action[1]))

if __name__ == '__main__':
    N = 6

    start = BlockWorldHeuristic(N)
    goal = BlockWorldHeuristic(N)

    print("Searching for a path:")
    print(f"{format_state(start.get_state())} -> {format_state(goal.get_state())}")
    print()

    astar = AStar()
    path = astar.search(start, goal)

    if path is not None:
        print("Found a path:")
        print([format_action(a) for a in path])

        print("\nHere's how it goes:")

        s = start.clone()
        print(format_state(s.get_state()))

        for a in path:
            s.apply(a)
            print(format_state(s.get_state()))

    else:
        print("No path exists.")

    print("Total expanded nodes:", BlockWorld.expanded)