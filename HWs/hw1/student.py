from blockworld import BlockWorld
from queue import PriorityQueue

class BlockWorldHeuristic(BlockWorld):
    def __init__(self, num_blocks=5, state=None):
        BlockWorld.__init__(self, num_blocks, state)

    def heuristic(self, goal):
        self_state = self.get_state()
        goal_state = goal.get_state()
        
        goal_below = {}
        for stack in goal_state:
            for i, block in enumerate(stack):
                goal_below[block] = stack[i-1] if i > 0 else None

        current_below = {}
        for stack in self_state:
            for i, block in enumerate(stack):
                current_below[block] = stack[i-1] if i > 0 else None

        def is_correct(block):
            support = current_below.get(block)
            if goal_below[block] == support:
                if support is None:
                    return True
                return is_correct(support)
            else:
                return False

        h = 0
        for stack in self_state:
            for block in stack:
                if not is_correct(block):
                    h += 1

        return float(h)

class AStar():
    def search(self, start, goal):

        queue = PriorityQueue()

        start_h = start.heuristic(goal)

        queue.put((start_h, 0, id(start), start, []))

        visited = {start.get_state(): 0}

        while not queue.empty():

            f, g, _, current, path = queue.get()

            if current == goal:
                return path

            for action, neighbor in current.get_neighbors():

                new_g = g + 1
                state = neighbor.get_state()

                if state not in visited or new_g < visited[state]:

                    visited[state] = new_g

                    h = neighbor.heuristic(goal)
                    f = new_g + h

                    queue.put((f, new_g, id(neighbor), neighbor, path + [action]))

        return None

if __name__ == '__main__':
	# Here you can test your algorithm. You can try different N values, e.g. 6, 7.
	N = 5

	start = BlockWorldHeuristic(N)
	goal = BlockWorldHeuristic(N)

	print("Searching for a path:")
	print(f"{start} -> {goal}")
	print()

	astar = AStar()
	path = astar.search(start, goal)

	if path is not None:
		print("Found a path:")
		print(path)

		print("\nHere's how it goes:")

		s = start.clone()
		print([[int(b) for b in stack] for stack in s.get_state()])

		for a in path:
			s.apply(a)
			print([[int(b) for b in stack] for stack in s.get_state()])

	else:
		print("No path exists.")

	print("Total expanded nodes:", BlockWorld.expanded)