import torch
import math
from src.optimization.func_wrapped import func_wrapped_parallel

class Interval():
    def __init__(self, size:float, center_value:float, bounds:list[tuple[float, float]], center_point: list[float]):
        """
        An interval in the optimization process.
        Args:
            size: diameter of the current interval, infinity norm of the bounds
            center_value: function value at the center point of the bounds of the current interval.
            bounds: List of tuples representing the bounds for each parameter.
            center_point: Coordinates of the center point representing the current interval.
            inf_pos: tuple, (pos, inf_num), inference_position. Position of the calculated center value in batched evaluation, for reproducibility only. If max_batch_num
            is 1 then just ignore this parameter.
        """
        self.size = size
        self.center_value = center_value
        self.bounds = bounds
        self.center_point = center_point
        # split_counter: list of integers representing how many times each dimension has been split.
        self.split_counter = [0] * len(bounds)
        self.inf_pos = (0, 1)

    def get_longest_side_index_all(self):
        """
        Get the list of index of the longest side of the bounds.
        If multiple sides are the longest, return the one with least splits.
        If still multiple, return the smallest index one.   
        Returns:
            index_list: list[int], list of index of the longest side
        """
        lengths = [b[1] - b[0] for b in self.bounds]
        max_length = max(lengths)
        candidate_indices = [i for i, l in enumerate(lengths) if l == max_length]
        if len(candidate_indices) == 1:
            return [candidate_indices[0]]
        else:
            # min_splits = min([self.split_counter[i] for i in candidate_indices])
            # final_candidates = [i for i in candidate_indices if self.split_counter[i] == min_splits]
            # sort the candidate_indices by split_counter, then by index
            final_candidates = sorted(candidate_indices, key=lambda i: (self.split_counter[i], i))
            return final_candidates
        
    def get_longest_side_index_single(self):
        """
        Get the index of the longest side of the bounds.
        If multiple sides are the longest, return the one with least splits.
        If still multiple, return the smallest index one.   
        Returns:
            index_list: list[int], index of the longest side, containing only one element
        """
        lengths = [b[1] - b[0] for b in self.bounds]
        max_length = max(lengths)
        candidate_indices = [i for i, l in enumerate(lengths) if l == max_length]
        if len(candidate_indices) == 1:
            return [candidate_indices[0]]
        else:
            min_splits = min([self.split_counter[i] for i in candidate_indices])
            final_candidates = [i for i in candidate_indices if self.split_counter[i] == min_splits]
            return final_candidates[:1]

    def update_size(self):
        """
        Update the size of the interval.
        """
        max_size = 0
        for b in self.bounds:
            max_size = max(max_size, b[1] - b[0])
        self.size = max_size

    def get_length(self, index):
        """
        Get the length of a specific side of the bounds.
        Args:
            index: int, index of the side
        Returns:
            length: float, length of the side
        """
        return self.bounds[index][1] - self.bounds[index][0]

    def get_distance(self, norm=2):
        """
        Get the distance of the interval: from center point to the corner point.
        Args:
            norm: int, norm to use
        Returns:
            distance: float, distance of the interval
        """
        return torch.norm(torch.tensor([b[1] - c for b, c in zip(self.bounds, self.center_point)]), p=norm).item()

    def get_volume(self):
        """
        Get the volume of the interval.
        Returns:
            volume: float, volume of the interval
        """
        volume = 1.0
        for b in self.bounds:
            volume *= (b[1] - b[0])
        return volume

        

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

def trisect_interval_pareto(interval:Interval, bounds, f, **kwargs):
    """
    Trisect an interval into child intervals.
    Following the pareto DIRECT algorithm
    Input:
        interval: Interval, the interval to be trisected
        bounds: list of tuples, [(min1, max1), (min2, max2), ...]
        f: function to minimize, which takes in a list of parameter lists and returns a list
        **kwargs: additional keyword arguments to pass to the function
    Return:
        child_interval_list: list[Interval], the child intervals after trisection. REMIND: the center_value is None and needs to be evaluated.
    """
    dim_to_split_list = interval.get_longest_side_index_single()

    to_evaluate_points = []

    for i in range(len(dim_to_split_list)):
        dim = dim_to_split_list[i]
        delta = interval.get_length(dim) / 3
        theta1 = interval.center_point.copy()
        theta1[dim] = interval.center_point[dim] - delta
        theta2 = interval.center_point.copy()
        theta2[dim] = interval.center_point[dim] + delta
        to_evaluate_points.append((theta1, theta2, dim))
        # (left_point, right_point, dim)

    theta_list = []
    for theta1, theta2, dim in to_evaluate_points:
        theta_list.append(projection(theta1, bounds))
        theta_list.append(projection(theta2, bounds))
    values = f(theta_list, parallel_num=len(theta_list), **kwargs)

    for i in range(len(to_evaluate_points)):
        to_evaluate_points[i] = (values[2 * i], values[2 * i + 1], to_evaluate_points[i][2])


    child_list = []
    for i, dim_to_split in enumerate(dim_to_split_list):
        length = interval.get_length(dim_to_split)
        child_bounds = interval.bounds.copy()
        # get right child
        sign = 1
        child_bounds[dim_to_split] = (
            interval.bounds[dim_to_split][0],
            interval.bounds[dim_to_split][0] + length / 3 * sign,
        )
        child_center_point = interval.center_point.copy()
        child_center_point[dim_to_split] = interval.center_point[dim_to_split] + length / 3 * sign
        child = Interval(
                            size=0,
                            center_value=to_evaluate_points[i][1],
                            bounds=child_bounds,
                            center_point=child_center_point,
                        )
        child.split_counter = [0] * len(child_bounds)
        child.update_size()
        child.inf_pos = (2 * i + 1, 2 * len(to_evaluate_points))
        child_list.append(child)

        # get left child
        sign = -1
        child_bounds = interval.bounds.copy()
        child_bounds[dim_to_split] = (
            interval.bounds[dim_to_split][1] + length / 3 * sign,
            interval.bounds[dim_to_split][1],
        )

        child_center_point = interval.center_point.copy()
        child_center_point[dim_to_split] = interval.center_point[dim_to_split] + length / 3 * sign
        child = Interval(
                            size=0,
                            center_value=to_evaluate_points[i][0],
                            bounds=child_bounds,
                            center_point=child_center_point,
                        )
        child.split_counter = [0] * len(child_bounds)
        child.update_size()
        child.inf_pos = (2 * i, 2 * len(to_evaluate_points))
        child_list.append(child)

        # update the current interval
        interval.bounds[dim_to_split] = (
            interval.bounds[dim_to_split][0] + length / 3,
            interval.bounds[dim_to_split][1] - length / 3,
        )
        interval.split_counter[dim_to_split] += 1
        interval.update_size()

    return child_list

def select_PLO(interval_list:list[Interval]):
    """
    Select the Pareto Liptchitzian Optimal (PLO) intervals from a list of intervals.
    An interval is PLO if no other interval dominates it.
    An interval A dominates interval B if:
        1. A.center_value < B.center_value and A.size > B.size or
        2. A.center_value < B.center_value and A.size == B.size or
        3. A.center_value == B.center_value and A.size > B.size
    The entire algorithm is O(n log n) where n is the number of intervals.

    Input:
        interval_list: list of Interval, the list of intervals to select from
    Return:
        PLO_list: list of Interval, the list of PLO intervals
    """
    x = [interval.size for interval in interval_list]
    y = [interval.center_value for interval in interval_list]
    index = list(range(len(interval_list)))
    points = list(zip(x, y, index))

    # remove duplicates with same (x, y) pair, keep the one with smallest idx
    unique_points = {}
    for (x, y, idx) in points:
        if (x, y) not in unique_points or idx < unique_points[(x, y)][2]:
            unique_points[(x, y)] = (x, y, idx)
    points = list(unique_points.values())

    # sort points by increasing order of x, then by decreasing order of y
    points.sort(key=lambda p: (p[0], -p[1]))
    # traverse the points in reverse order, keep track of the current minimun y, and drop those points with y larger than the current minimum y
    PLO_indices = []
    current_min_y = float('inf')
    for i in range(len(points)-1, -1, -1):
        x, y, idx = points[i]
        if y < current_min_y:
            PLO_indices.append(interval_list[idx])
            current_min_y = y
    return PLO_indices

@torch.no_grad()
def pareto_direct_old(bounds, f_parallel, max_iter=1000, **kwargs):
    """
    Simple_direct optimization
    Input:
        bounds: list of tuples, [(min1, max1), (min2, max2), ...]
        f_parallel: function to minimize, which takes in a list of parameter lists and returns a list of function values(could be parallelized)
        max_iter: maximum number of iterations
        kwargs: additional arguments to pass to f_parallel

    Return:
        (best_value, best_params, value_list, inf_pos)
        where best_value is the minimum value found and best_params are the parameters that give this value
        and value_list is the list of best values found at each improvement step
        And inf_pos is the position of the best_value in batched evaluation, tuple: (pos, max_pos), for reproducibility only. If max_batch_num
        is 1 then just ignore this parameter.
    """
    # Initialize
    num_params = len(bounds)
    best_value = float('inf')
    best_params = None
    inf_pos = (0, 1)
    eval_count = 0
    value_list = []


    # Create the initial interval, covering the whole search hybercube [0, 1]^n
    initial_interval = Interval(
        size=1,
        center_value=f_parallel(
            [projection([0.5] * num_params, bounds)],
            parallel_num=1,
            **kwargs,
        )[0],
        bounds=[(0.0, 1.0)] * num_params,
        center_point=[0.5] * num_params,
    )
    eval_count += 1
    # update best value and params
    best_value = initial_interval.center_value
    best_params = projection(initial_interval.center_point, bounds)
    inf_pos = initial_interval.inf_pos
    value_list.append(best_value)

    interval_list = [initial_interval]
    PLO_list = [initial_interval]

    # iteration
    iter_time = 0
    while iter_time < max_iter:
        # free cuda memory
        # torch.cuda.empty_cache()
        # split each interval in PLO_list
        new_interval_list = []
        for interval in PLO_list:
            # for pareto DIRECT, use trisect_interval_pareto
            new_intervals = trisect_interval_pareto(interval, bounds, f_parallel, **kwargs)
            new_interval_list += new_intervals

        for interval in new_interval_list:
            eval_count += 1
            value = interval.center_value
            # update best value and params
            if value < best_value:
                best_value = value
                best_params = projection(interval.center_point, bounds)
                if interval.inf_pos[1] < interval.inf_pos[1] // kwargs["max_batch_num"] * kwargs["max_batch_num"]:
                    inf_pos = (interval.inf_pos[0] % kwargs["max_batch_num"], kwargs["max_batch_num"])
                else:
                    inf_pos = (interval.inf_pos[0] % kwargs["max_batch_num"], interval.inf_pos[1] % kwargs["max_batch_num"] + 1)

        interval_list += new_interval_list
        # early stopping if the interval containing the best_value is too small
        best_count, smaller_len_count, smaller_vol_count = 0, 0, 0
        for interval in interval_list:
            if interval.center_value == best_value:
                best_count += 1
                if interval.get_volume() < 1e-16:
                    return best_value, best_params, value_list, inf_pos
                    smaller_vol_count += 1
                if interval.get_length(interval.get_longest_side_index_single()[0]) < 1e-6:
                    smaller_len_count += 1
        if best_count > 0 and max(smaller_len_count / best_count, smaller_vol_count / best_count) >= 0.5:
            return best_value, best_params, value_list, inf_pos
        
        # select PLO_list from interval_list, pareto DIRECT
        PLO_list = select_PLO(interval_list)

        value_list.append(best_value)
        iter_time += 1
        # print(f"Iteration {iter_time}: Best value = {best_value}, Evaluations = {eval_count}, PLO count = {len(PLO_list)}")

    return best_value, best_params, value_list, inf_pos


@torch.no_grad()
def original_direct(bounds, f_parallel, max_iter=1000, **kwargs):
    """
    Original_direct optimization
    Input:
        bounds: list of tuples, [(min1, max1), (min2, max2), ...]
        f_parallel: function to minimize, which takes in a list of parameter lists and returns a list of function values(could be parallelized)
        max_iter: maximum number of iterations
        kwargs: additional arguments to pass to f_parallel

    Return:
        (best_value, best_params, value_list, inf_pos)
        where best_value is the minimum value found and best_params are the parameters that give this value
        and value_list is the list of best values found at each improvement step
        And inf_pos is the position of the best_value in batched evaluation, tuple: (pos, max_pos), for reproducibility only. If max_batch_num
        is 1 then just ignore this parameter.
    """
    # Initialize
    num_params = len(bounds)
    best_value = float('inf')
    best_params = None
    inf_pos = (0, 1)
    eval_count = 0
    value_list = []


    # Create the initial interval, covering the whole search hybercube [0, 1]^n
    initial_interval = Interval(
        size=1,
        center_value=f_parallel(
            [projection([0.5] * num_params, bounds)],
            parallel_num=1,
            **kwargs,
        )[0],
        bounds=[(0.0, 1.0)] * num_params,
        center_point=[0.5] * num_params,
    )
    eval_count += 1
    # update best value and params
    best_value = initial_interval.center_value
    best_params = projection(initial_interval.center_point, bounds)
    inf_pos = initial_interval.inf_pos

    value_list.append(best_value)

    interval_list = [initial_interval]
    PTO_list = [initial_interval]

    # iteration
    iter_time = 0
    while iter_time < max_iter:
        # free cuda memory
        # torch.cuda.empty_cache()
        # split each interval in PLO_list
        new_interval_list = []
        for interval in PTO_list:
            # for standard DIRECT, use trisect_interval_DIRECT
            new_intervals = trisect_interval_DIRECT(interval, bounds, f_parallel, **kwargs)
            new_interval_list += new_intervals

        for interval in new_interval_list:
            eval_count += 1
            value = interval.center_value
            # update best value and params
            if value < best_value:
                best_value = value
                best_params = projection(interval.center_point, bounds)
                if interval.inf_pos[1] < interval.inf_pos[1] // kwargs["max_batch_num"] * kwargs["max_batch_num"]:
                    inf_pos = (interval.inf_pos[0] % kwargs["max_batch_num"], kwargs["max_batch_num"])
                else:
                    inf_pos = (interval.inf_pos[0] % kwargs["max_batch_num"], interval.inf_pos[1] % kwargs["max_batch_num"] + 1)

        interval_list += new_interval_list

        # early stopping if the interval containing the best_value is too small
        best_count, smaller_len_count, smaller_vol_count = 0, 0, 0
        for interval in interval_list:
            if interval.center_value == best_value:
                best_count += 1
                if interval.get_volume() < 1e-16:
                    return best_value, best_params, value_list, inf_pos
                    smaller_vol_count += 1
                if interval.get_length(interval.get_longest_side_index_single()[0]) < 1e-6:
                    smaller_len_count += 1
        if best_count > 0 and max(smaller_len_count / best_count, smaller_vol_count / best_count) >= 0.5:
            # # debug
            # print("iter_time:", iter_time)
            # print("best_count:", best_count)
            # print("smaller_len_count:", smaller_len_count)
            # print("smaller_vol_count:", smaller_vol_count)

            return best_value, best_params, value_list, inf_pos

        # select PTO_list from interval_list, standard DIRECT
        PTO_list = select_PTO(interval_list, best_value, epsilon=1e-4)

        value_list.append(best_value)
        iter_time += 1
        # print(f"Iteration {iter_time}: Best value = {best_value}, Evaluations = {eval_count}, PTO count = {len(PTO_list)}")

    return best_value, best_params, value_list, inf_pos


def select_PTO(interval_list:list[Interval], current_best_value:float=None, epsilon:float=1e-4):
    """
    Select the PoTential Optimal (PTO) intervals from a list of intervals, that is, the lower convex hull of the points (size, center_value).
    An interval is PTO if no other interval dominates it.
    The entire algorithm is O(n log n) where n is the number of intervals.

    Input:
        interval_list: list of Interval, the list of intervals to select from
    Return:
        PTO_list: list of Interval, the list of PLO intervals
    """
    from scipy.spatial import ConvexHull, QhullError
    x = [interval.size for interval in interval_list]
    y = [interval.center_value for interval in interval_list]

    # if all x are the same, return the one with smallest y
    if max(x) == min(x):
        min_index = y.index(min(y))
        return [interval_list[min_index]]
    
    index = list(range(len(interval_list)))
    points = list(zip(x, y, index))

    # filter those which do not satisfy:
    # f - K * d <= f_min - epsilon * |f_min|
    # by selecting the first point met when rotating a line acrossing point (0, f_min - epsilon * |f_min|) anticlockwise
    anchor = (0, current_best_value - epsilon * abs(current_best_value))
    first_angle = float('inf')
    first_point = []
    for p in points:
        vx, vy = p[0] - anchor[0], p[1] - anchor[1]
        angle = math.atan2(vy, vx)
        if angle < first_angle:
            first_angle = angle
            first_point = p 
    # filter out points that has a smaller x coordinate than the first point
    for p in points:
        if p[0] < first_point[0]:
            points.remove(p)
    
    # remove duplicates with same (x, y) pair, keep the one with smallest idx
    unique_points = {}
    for (x, y, idx) in points:
        if (x, y) not in unique_points or idx < unique_points[(x, y)][2]:
            unique_points[(x, y)] = (x, y, idx)

    points = [(p[0], p[1]) for p in unique_points.values()]
    try:
        hull = ConvexHull(points)
    except QhullError:
        hull = ConvexHull(points, qhull_options="QJ")

    hull_points = [list(unique_points.values())[v] for v in hull.vertices]

    left = min(hull_points, key=lambda p: (p[0], p[1]))
    right = max(hull_points, key=lambda p: (p[0], p[1]))

    li = hull_points.index(left)
    ri = hull_points.index(right)

    path1 = hull_points[li:ri+1]
    path2 = hull_points[ri:] + hull_points[:li+1]

    # choose the path that has lower y values
    path1_y = sum([p[1] for p in path1])
    path2_y = sum([p[1] for p in path2])
    if path1_y < path2_y:
        PTO_list = [interval_list[p[2]] for p in path1]
    else:
        PTO_list = [interval_list[p[2]] for p in path2]

    # # if enable TIEs:
    # # those has same (x, y) as any point in PTO_list should also be included
    # for interval in interval_list:
    #     for p in PTO_list:
    #         if interval.size == p.size and interval.center_value == p.center_value and interval not in PTO_list:
    #             PTO_list.append(interval)
    #             break

    return PTO_list

def trisect_interval_DIRECT(interval:Interval, bounds, f, **kwargs):
    """
    Trisect an interval into child intervals.
    Following the original DIRECT algorithm, trisect the dimension from lowest wk to highest, where wk = min(f(center - \delta * e_k), f(center + \delta * e_k))
    And the f(center \pm \delta * e_k) could be used for new intervals, so their values should be cached. That's why we put merge them in trisect function.
    Input:
        interval: Interval, the interval to be trisected
        bounds: list of tuples, [(min1, max1), (min2, max2), ...]
        f: function to minimize, which takes in a list of parameter lists and returns a list
        **kwargs: additional keyword arguments to pass to the function
    Return:
        child_interval_list: list[Interval], the child intervals after trisection. REMIND: the center_value is None and needs to be evaluated.
    """
    dim_to_split_list = interval.get_longest_side_index_all()
    child_list = []
    # sort the dim_to_split_list by increasing order of wk = min(f(center - \delta * e_k), f(center + \delta * e_k))
    to_evaluate_points = []

    for i in range(len(dim_to_split_list)):
        dim = dim_to_split_list[i]
        delta = interval.get_length(dim) / 3
        theta1 = interval.center_point.copy()
        theta1[dim] = interval.center_point[dim] - delta
        theta2 = interval.center_point.copy()
        theta2[dim] = interval.center_point[dim] + delta
        to_evaluate_points.append((theta1, theta2, dim))
        # (left_point, right_point, dim)

    theta_list = []
    for theta1, theta2, dim in to_evaluate_points:
        theta_list.append(projection(theta1, bounds))
        theta_list.append(projection(theta2, bounds))
    values = f(theta_list, parallel_num=len(theta_list), **kwargs)

    for i in range(len(to_evaluate_points)):
        to_evaluate_points[i] = (values[2 * i], values[2 * i + 1], to_evaluate_points[i][2])        

    # sort by increasing order of wk
    to_evaluate_points.sort(key=lambda x: min(x[0], x[1]))
    dim_to_split_list = [x[2] for x in to_evaluate_points]
    
    for i, dim_to_split in enumerate(dim_to_split_list):
        length = interval.get_length(dim_to_split)
        child_bounds = interval.bounds.copy()
        # get right child
        sign = 1
        child_bounds[dim_to_split] = (
            interval.bounds[dim_to_split][0],
            interval.bounds[dim_to_split][0] + length / 3 * sign,
        )
        child_center_point = interval.center_point.copy()
        child_center_point[dim_to_split] = interval.center_point[dim_to_split] + length / 3 * sign
        # get right child's cached center value
        child = Interval(
                            size=0,
                            center_value=to_evaluate_points[i][1],
                            bounds=child_bounds,
                            center_point=child_center_point,
                        )
        child.split_counter = [0] * len(child_bounds)
        child.update_size()
        child.inf_pos = (2 * i + 1, 2 * len(to_evaluate_points))
        child_list.append(child)

        # get left child
        sign = -1
        child_bounds = interval.bounds.copy()
        child_bounds[dim_to_split] = (
            interval.bounds[dim_to_split][1] + length / 3 * sign,
            interval.bounds[dim_to_split][1],
        )

        child_center_point = interval.center_point.copy()
        child_center_point[dim_to_split] = interval.center_point[dim_to_split] + length / 3 * sign
        child = Interval(
                            size=0,
                            center_value=to_evaluate_points[i][0],
                            bounds=child_bounds,
                            center_point=child_center_point,
                        )
        child.split_counter = [0] * len(child_bounds)
        child.update_size()
        child.inf_pos = (2 * i, 2 * len(to_evaluate_points))
        child_list.append(child)

        # update the current interval
        interval.bounds[dim_to_split] = (
            interval.bounds[dim_to_split][0] + length / 3,
            interval.bounds[dim_to_split][1] - length / 3,
        )
        interval.split_counter[dim_to_split] += 1
        interval.update_size()

    return child_list



@torch.no_grad()
def pareto_direct(bounds, f_parallel, max_iter=1000, **kwargs):
    """
    Simple_direct optimization
    Input:
        bounds: list of tuples, [(min1, max1), (min2, max2), ...]
        f_parallel: function to minimize, which takes in a list of parameter lists and returns a list of function values(could be parallelized)
        max_iter: maximum number of iterations
        kwargs: additional arguments to pass to f_parallel

    Return:
        (best_value, best_params, value_list, inf_pos)
        where best_value is the minimum value found and best_params are the parameters that give this value
        and value_list is the list of best values found at each improvement step
        And inf_pos is the position of the best_value in batched evaluation, tuple: (pos, max_pos), for reproducibility only. If max_batch_num
        is 1 then just ignore this parameter.
    """
    # Initialize
    num_params = len(bounds)
    best_value = float('inf')
    best_params = None
    inf_pos = (0, 1)
    eval_count = 0
    value_list = []


    # Create the initial interval, covering the whole search hybercube [0, 1]^n
    initial_interval = Interval(
        size=1,
        center_value=f_parallel(
            [projection([0.5] * num_params, bounds)],
            parallel_num=1,
            **kwargs,
        )[0],
        bounds=[(0.0, 1.0)] * num_params,
        center_point=[0.5] * num_params,
    )
    eval_count += 1
    # update best value and params
    best_value = initial_interval.center_value
    best_params = projection(initial_interval.center_point, bounds)
    inf_pos = initial_interval.inf_pos
    value_list.append(best_value)

    interval_list = [initial_interval]
    PLO_list = [initial_interval]

    # iteration
    iter_time = 0
    while iter_time < max_iter:
        # free cuda memory
        # torch.cuda.empty_cache()
        # split each interval in PLO_list
        new_interval_list = []
        to_evaluate_points = []
        father_interval = []

        theta_list = []
        for interval in PLO_list:
            # new_intervals = trisect_interval_pareto(interval, bounds, f_parallel, **kwargs)
            dim_to_split_list = interval.get_longest_side_index_single()

            for i in range(len(dim_to_split_list)):
                dim = dim_to_split_list[i]
                delta = interval.get_length(dim) / 3
                theta1 = interval.center_point.copy()
                theta1[dim] = interval.center_point[dim] - delta
                theta2 = interval.center_point.copy()
                theta2[dim] = interval.center_point[dim] + delta
                to_evaluate_points.append((theta1, theta2, dim))
                # (left_point, right_point, dim)
                father_interval.append(interval)

        for theta1, theta2, dim in to_evaluate_points:
            theta_list.append(projection(theta1, bounds))
            theta_list.append(projection(theta2, bounds))

        values = f_parallel(theta_list, parallel_num=len(theta_list), **kwargs)

        for i in range(len(to_evaluate_points)):
            to_evaluate_points[i] = (values[2 * i], values[2 * i + 1], to_evaluate_points[i][2])

        child_list = []
        for idx, interval in enumerate(father_interval):
            dim_to_split_list = interval.get_longest_side_index_single()
            for dim_to_split in dim_to_split_list:
                length = interval.get_length(dim_to_split)
                child_bounds = interval.bounds.copy()
                # get right child
                sign = 1
                child_bounds[dim_to_split] = (
                    interval.bounds[dim_to_split][0],
                    interval.bounds[dim_to_split][0] + length / 3 * sign,
                )
                child_center_point = interval.center_point.copy()
                child_center_point[dim_to_split] = interval.center_point[dim_to_split] + length / 3 * sign
                child = Interval(
                                    size=0,
                                    center_value=to_evaluate_points[idx][1],
                                    bounds=child_bounds,
                                    center_point=child_center_point,
                                )
                child.split_counter = [0] * len(child_bounds)
                child.update_size()
                child.inf_pos = (2 * idx + 1, 2 * len(to_evaluate_points))
                child_list.append(child)

                # get left child
                sign = -1
                child_bounds = interval.bounds.copy()
                child_bounds[dim_to_split] = (
                    interval.bounds[dim_to_split][1] + length / 3 * sign,
                    interval.bounds[dim_to_split][1],
                )

                child_center_point = interval.center_point.copy()
                child_center_point[dim_to_split] = interval.center_point[dim_to_split] + length / 3 * sign
                child = Interval(
                                    size=0,
                                    center_value=to_evaluate_points[idx][0],
                                    bounds=child_bounds,
                                    center_point=child_center_point,
                                )
                child.split_counter = [0] * len(child_bounds)
                child.update_size()
                child.inf_pos = (2 * idx, 2 * len(to_evaluate_points))
                child_list.append(child)

                # update the current interval
                interval.bounds[dim_to_split] = (
                    interval.bounds[dim_to_split][0] + length / 3,
                    interval.bounds[dim_to_split][1] - length / 3,
                )
                interval.split_counter[dim_to_split] += 1
                interval.update_size()

        new_interval_list += child_list

        for interval in new_interval_list:
            eval_count += 1
            value = interval.center_value
            # update best value and params
            if value < best_value:
                best_value = value
                best_params = projection(interval.center_point, bounds)
                if interval.inf_pos[1] < interval.inf_pos[1] // kwargs["max_batch_num"] * kwargs["max_batch_num"]:
                    inf_pos = (interval.inf_pos[0] % kwargs["max_batch_num"], kwargs["max_batch_num"])
                else:
                    inf_pos = (interval.inf_pos[0] % kwargs["max_batch_num"], interval.inf_pos[1] % kwargs["max_batch_num"] + 1)

        interval_list += new_interval_list
        # early stopping if the interval containing the best_value is too small
        best_count, smaller_len_count, smaller_vol_count = 0, 0, 0
        for interval in interval_list:
            if interval.center_value == best_value:
                best_count += 1
                if interval.get_volume() < 1e-16:
                    return best_value, best_params, value_list, inf_pos
                    smaller_vol_count += 1
                if interval.get_length(interval.get_longest_side_index_single()[0]) < 1e-6:
                    smaller_len_count += 1
        if best_count > 0 and max(smaller_len_count / best_count, smaller_vol_count / best_count) >= 0.5:
            return best_value, best_params, value_list, inf_pos
        
        # select PLO_list from interval_list, pareto DIRECT
        PLO_list = select_PLO(interval_list)

        value_list.append(best_value)
        iter_time += 1
        # print(f"Iteration {iter_time}: Best value = {best_value}, Evaluations = {eval_count}, PLO count = {len(PLO_list)}")

    return best_value, best_params, value_list, inf_pos
