class AStarNode:
    def __init__(self, pos, parent, g, h):
        self.pos = pos
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent

    def update(self, new_g, parent):
        self.g = new_g
        self.f = self.g + self.h
        self.parent = parent


def pathfind(grid, start_pos, target_pos, obstacles=(3,)):
    open_list = list()
    closed_list = list()

    # Add start node to list
    start_node = AStarNode(start_pos, None, 0, pythagoras_distance(start_pos, target_pos))
    open_list.append(start_node)

    while True:
        # If the open list is empty, there is no path. Return None.
        if len(open_list) == 0:
            return None

        # Find the lowest cost square on the open list
        current_node = min(open_list, key=lambda n: n.f)

        # If node is target, retrace path
        if current_node.pos == target_pos:
            path = list()
            while True:
                path.append(current_node.pos)
                if current_node.parent is None:
                    path.reverse()
                    return path
                current_node = current_node.parent

        open_list.remove(current_node)
        closed_list.append(current_node)

        # Scan all the nodes adjacent to the current node
        for y in range(current_node.pos[1] - 1, current_node.pos[1] + 2):
            for x in range(current_node.pos[0] - 1, current_node.pos[0] + 2):
                pos = (x, y)
                # Check if walkable or in closed list
                if x < 0 or y < 0 or x >= len(grid[0]) or y >= len(grid):
                    continue
                if grid[y][x] in obstacles or pos_in_list(pos, closed_list):
                    continue

                node = get_node_in_list(pos, open_list)
                if node is not None:
                    # Checks if the current node provides a better path
                    if node.g < current_node.g + 1:
                        node.update(current_node.g + 1, current_node)
                else:
                    # Adds the node to the open list if not existing
                    node = AStarNode(pos,
                                     current_node,
                                     current_node.g + 1,
                                     pythagoras_distance(pos, target_pos))
                    open_list.append(node)


def pythagoras_distance(pos_a, pos_b):
    return pow((pos_a[0] - pos_b[0]), 2) + pow((pos_a[1] - pos_b[1]), 2)


def pos_in_list(pos, node_list):
    for node in node_list:
        if node.pos[0] == pos[0] and node.pos[1] == pos[1]:
            return True

    return False


def get_node_in_list(pos, node_list):
    for node in node_list:
        if node.pos[0] == pos[0] and node.pos[1] == pos[1]:
            return node
    return None
