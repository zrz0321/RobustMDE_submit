import torch

class Node():
    def __init__(self, size:float, depth:int, center_value:float, bounds:list[tuple[float, float]], center_point: list[float], local_slope:float=None):
        """
        A node in the optimization tree.
        Args:
            size: diameter of the current node, infinity norm of the bounds
            depth: Current depth in the tree.
            cemter_value: function value at the center point of the bounds of the current node.
            bounds: List of tuples representing the bounds for each parameter.
            local_slope: Local slope for the current node.
        """
        self.size = size
        self.depth = depth
        self.center_value = center_value
        self.bounds = bounds
        self.center_point = center_point
        self.local_slope = local_slope

    
    def get_longest_side_index(self):
        """
        Get the index of all the longest side of the bounds.
        Returns:
            index: int, index of the longest side
        """
        lengths = [b[1] - b[0] for b in self.bounds]
        max_length = max(lengths)
        return [i for i, l in enumerate(lengths) if l == max_length]
    
    def update_size_and_depth(self):
        """
        Update the size and depth of the node.
        """
        max_size = 0
        for b in self.bounds:
            max_size = max(max_size, b[1] - b[0])
        self.size = max_size
        h = 0
        if max_size == 1:
            self.depth = 0
        else:
            while max_size < 1 / (3 ** h):
                h += 1
            self.depth = h

def projection(theta, bounds):
    """
    Project a normalized parameter value to its original range.
    Args:
        theta: List of normalized parameter values in [0, 1].
        bounds: List of tuples representing the bounds for each parameter.
    Returns:
        projected_theta: List of parameter values in their original range.
    """
    return [b[0] + (b[1] - b[0]) * t for t, b in zip(theta, bounds)]


def simple_direct(bounds, f, max_iter=1000, max_depth=10, **kwargs):
    """
    Simple_direct optimization
    Input:
        bounds: list of tuples, [(min1, max1), (min2, max2), ...]
        f: function to minimize
        max_iter: maximum number of iterations
        kwargs: additional arguments to pass to f
    
    Return:
        (best_value, best_params) where best_value is the minimum value found and best_params are the parameters that give this value
    """

    def calculate_local_slope(center_point, center_value, depth, bounds):
        """
        Calculate the local slope for a node.
        Args:
            center_point: List of normalized parameter values in [0, 1].
            center_value: Function value at the center point.
            depth: Depth of the node in the tree.
        Returns:
            local_slope: Local slope for the node.
        """
        lengths = [b[1] - b[0] for b in bounds]
        max_length = max(lengths)
        m = [i for i, l in enumerate(lengths) if l == max_length]
        local_slope = -1
        for i in m:
            new_center_point_1 = center_point.copy()
            new_center_point_1[i] -= 1 / (3 ** (depth + 1))
            new_center_point_2 = center_point.copy()
            new_center_point_2[i] += 1 / (3 ** (depth + 1))
            f1 = f(projection(new_center_point_1, bounds), **kwargs)
            f2 = f(projection(new_center_point_2, bounds), **kwargs)
            current_slope = max(abs(center_value - f1), abs(center_value - f2)) * (3 ** (depth + 1))
            local_slope = max(local_slope, current_slope)
        return local_slope
        

    # Initialize
    num_params = len(bounds)
    best_value = float('inf')
    best_params = None
    min_size = 3 ** (-max_depth)
    PO_node = [Node(
        size=1,
        depth=0,
        center_value=f(projection([0.5] * num_params, bounds), **kwargs),
        bounds=[(0, 1) for _ in range(num_params)],
        center_point=[0.5] * num_params,
        local_slope=0,
    )]

    node_list = PO_node.copy()

    iter_time = 0
    while iter_time < max_iter and len(PO_node) > 0:
        X, dim_for_X = [], []
        for node in PO_node:
            if node.size > min_size:
                # EXPAND area
                m = node.get_longest_side_index()
                for i in m:
                    new_center_point_1 = node.center_point.copy()
                    new_center_point_1[i] -= node.size / 3
                    new_center_point_2 = node.center_point.copy()
                    new_center_point_2[i] += node.size / 3
                    X.append((new_center_point_1, new_center_point_2))
                    dim_for_X.append(i)
            Y_sort, Y = [], []
            for x in X:
                x1, x2 = x
                y1 = f(projection(x1, bounds), **kwargs)
                y2 = f(projection(x2, bounds), **kwargs)
                Y_sort.append(min(y1, y2))
                Y.append((y1, y2))

                if y1 < best_value:
                    best_value = y1
                    best_params = projection(x1, bounds)
                if y2 < best_value:
                    best_value = y2
                    best_params = projection(x2, bounds)
            
            order = sorted(range(len(Y_sort)), key=lambda i: Y[i])
            for idx in order:
                x1, x2 = X[idx]
                y1, y2 = Y[idx]
                i = dim_for_X[idx]
                # Trisect the i-th dimension
                new_bounds_1 = node.bounds.copy()
                new_bounds_1[i] = (node.bounds[i][0], node.bounds[i][0] + node.size / 3)
                new_bounds_2 = node.bounds.copy()
                new_bounds_2[i] = (node.bounds[i][1] - node.size / 3, node.bounds[i][1])
                new_node_1 = Node(
                    size=0,
                    depth=0,
                    center_value=y1,
                    bounds=new_bounds_1,
                    center_point=x1,
                )
                new_node_1.update_size_and_depth()
                new_node_2 = ...
                pass




            
