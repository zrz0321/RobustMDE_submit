from scipy.optimize import direct, Bounds, dual_annealing

def direct_optimization(bounds, f, max_iter=1000, *args):
    """
    Perform DIRECT optimization using scipy.optimize.direct to find the best perturbation parameters.
    INPUT:
    bounds: list of tuples, bounds for each parameter, e.g., [(min1, max1), (min2, max2), ...]
    f: wrapped function, the objective function to minimize
    *args: additional arguments to pass to the function f

    RETURN: 
    tuple: (best objective value, best parameters)
    """
    _bounds = Bounds(*zip(*bounds))
    result = direct(f, _bounds, maxiter=max_iter, args=args)
    return result.fun, result.x

def dual_annealing_optimization(bounds, f, max_iter=1000, *args):
    """
    Perform dual annealing optimization using scipy.optimize.dual_annealing to find the best perturbation parameters.
    INPUT:
    bounds: list of tuples, bounds for each parameter, e.g., [(min1, max1), (min2, max2), ...]
    f: wrapped function, the objective function to minimize
    *args: additional arguments to pass to the function f

    RETURN: 
    tuple: (best objective value, best parameters)
    """
    result = dual_annealing(f, bounds, maxiter=max_iter, args=args)
    return result.fun, result.x