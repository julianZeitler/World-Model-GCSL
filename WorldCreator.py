from scipy.ndimage import label
import numpy as np

def ensure_connected(world):
    """
    Ensures that the world is fully traversable by keeping only the largest connected component of free space.
    """
    labeled_array, num_features = label(world == 0)  # Label connected components of free space
    if num_features > 1:
        largest_component = np.argmax(np.bincount(labeled_array.flat)[1:]) + 1
        world[(labeled_array != largest_component) & (world == 0)] = 1  # Convert smaller regions to obstacles
    return world

def generate_world(seed=None):
    if seed is not None:
        np.random.seed(seed)

    world = np.ones((10, 10), dtype=int)  # Start with a world full of obstacles

    # Create paths by carving out sections
    world[1:-1, 1:-1] = 0  # Carve out a central traversable area

    # Add some clustered obstacles
    for _ in range(10):  # Adjust number of obstacles
        x, y = np.random.randint(1, 9), np.random.randint(1, 9)
        size_x, size_y = np.random.randint(1, 3), np.random.randint(1, 3)  # Small clusters
        world[x:x+size_x, y:y+size_y] = 1

    # Ensure the world is fully connected
    world = ensure_connected(world)
    
    return world
