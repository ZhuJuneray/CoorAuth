# from itertools import combinations

# def generate_edges(grid_size):
#     """ Generate all possible edges in a grid of given size. """
#     vertices = range(grid_size**2)
#     return list(combinations(vertices, 2))

# def pattern_to_edges(pattern, grid_size):
#     """ Convert a pattern of connected points to a set of edges. """
#     edges = set()
#     for i in range(len(pattern) - 1):
#         # Convert 2D grid coordinates to single number
#         start = pattern[i][0] * grid_size + pattern[i][1]
#         end = pattern[i + 1][0] * grid_size + pattern[i + 1][1]
#         edges.add((min(start, end), max(start, end)))
#     return edges

# def edit_distance(pattern1, pattern2, grid_size):
#     """ Calculate the edit distance between two patterns. """
#     edges1 = pattern_to_edges(pattern1, grid_size)
#     edges2 = pattern_to_edges(pattern2, grid_size)

#     # Edges to add or remove to transform pattern1 into pattern2
#     to_add = edges2 - edges1
#     to_remove = edges1 - edges2

#     # Edit distance is the total number of additions and deletions
#     return len(to_add) + len(to_remove)

# # Example patterns
# pattern1 = [(0, 0), (0, 1), (1, 1), (1, 2)]
# pattern2 = [(1, 2), (1, 1), (0, 1), (0, 0)]

# # Calculate edit distance for a 3x3 grid
# grid_size = 3
# distance = edit_distance(pattern1, pattern2, grid_size)
# print(distance)


def pattern_to_directed_edges(pattern, grid_size):
    """ Convert a pattern of connected points to a set of directed edges. """
    edges = set()
    for i in range(len(pattern) - 1):
        start = pattern[i][0] * grid_size + pattern[i][1]
        end = pattern[i + 1][0] * grid_size + pattern[i + 1][1]
        edges.add((start, end))  # Directed edge
    return edges

def edit_distance_directed(pattern1, pattern2, grid_size):
    """ Calculate the edit distance between two directed patterns. """
    edges1 = pattern_to_directed_edges(pattern1, grid_size)
    edges2 = pattern_to_directed_edges(pattern2, grid_size)

    to_add = edges2 - edges1
    to_remove = edges1 - edges2

    return len(to_add) + len(to_remove)

# Example directed patterns
pattern1 = [(0, 0), (0, 1), (1, 1), (1, 2)]
pattern2 = [(1, 2), (1, 1), (0, 1), (0, 0)]

# Calculate directed edit distance for a 3x3 grid
grid_size = 3
directed_distance = edit_distance_directed(pattern1, pattern2, grid_size)
print(directed_distance)
