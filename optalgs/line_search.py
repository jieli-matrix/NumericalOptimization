from warnings import warn
import numpy as np

class LineSearchWarning(RuntimeWarning):
    pass


def exact_line_search(f, grad_f, x_0, max_iter=500):
    """ Exact line search implemented for hw1

    Parameters
    ----------
    f :  callable f(x,*args)
        Objective function.
    grad_f : callable f'(x,*args)
        Objective function gradient direction (used for linear search).
    x_0 : ndarray
        Starting point.
    maxiter : int, optional
        Maximum number of iterations to perform.

    Returns
    -------
    alphas : float list or none
        Alpha for each step ``x_{k+1} = x_{k} + alpha * grad_fc``
        or None if the line search algorithm did not converge.
    fc : int
        Number of function evaluations made.
    new_fval : float or None
        New function value at end point,
        or None if the line search algorithm did not converge.
    """

    fc = 0
    x = x_0
    d = -np.array(grad_f(x))
    alphas = []
    while np.linalg.norm(d) > 1e-5:
        # in sepcial case you could get the argmin alpha in explicit expressions
        # but most of cases not.
        # don't recommend for exact line search
        alpha = (-d[0]*x[0] - 2*d[1]*x[1])/(d[0]**2 + 2*d[1]**2)
        alphas.append(alpha)
        # update x
        x = x + alpha*d
        # update d
        d = -np.array(grad_f(x))
        # update fc
        fc = fc + 1
        if fc >= max_iter:
            warn('The line search algorithm did not converge', LineSearchWarning)
            alphas = None
    new_fval = f(x)
    return alphas, fc, x, new_fval

def constant_step_search(f, grad_f, x_0, alpha, max_iter=500):
    """ Constant step line search implemented for hw1

    Parameters
    ----------
    f :  callable f(x,*args)
        Objective function.
    grad_f : callable f'(x,*args)
        Objective function gradient direction (used for linear search).
    alpha : float
        Constant alpha for each step
    x_0 : ndarray
        Starting point.
    maxiter : int, optional
        Maximum number of iterations to perform.
    
    Returns
    -------
    fc : int
        Number of function evaluations made.
    new_fval : float or None
        New function value at end point,
        or None if the line search algorithm did not converge.
    """
    fc = 0
    x = x_0
    d = -np.array(grad_f(x))
    while np.linalg.norm(d) > 1e-5:
        # update x
        x = x + alpha*d
        # update d
        d = -np.array(grad_f(x))
        # update fc
        fc = fc + 1
        if fc >= max_iter:
            warn('The line search algorithm did not converge', LineSearchWarning)
    new_fval = f(x)
    return fc, x, new_fval

def armijo_line_search(f, grad_f, x_0, alpha_0=1, gama=0.8, c=0.5, max_iter=500):
    """ Armijo line search implemented for hw1

    Parameters
    ----------
    f :  callable f(x,*args)
        Objective function.
    grad_f : callable f'(x,*args)
        Objective function gradient direction (used for linear search).
    x_0 : ndarray
        Starting point.
    alpha_0 : float, optional
        initial value for alpha
    gama : float in (0,1), optional
        decaying rate
    c : float in (0,1), optional
        scale
    maxiter : int, optional
        Maximum number of iterations to perform.

    Returns
    -------
    alphas : float list or none
        Alpha for each step ``x_{k+1} = x_{k} + alpha * grad_fc``
        or None if the line search algorithm did not converge.
    fc : int
        Number of function evaluations made.
    new_fval : float or None
        New function value at end point,
        or None if the line search algorithm did not converge.
    """

    fc = 0
    x = x_0
    d = -np.array(grad_f(x))
    alphas = []
    alpha = alpha_0
    while np.linalg.norm(d) > 1e-5:
        alpha = _alpha_search_armijo(f, grad_f, x, alpha, gama, c, d)
        # append alpha
        alphas.append(alpha)
        # update x to x_next
        x = x + alpha*d
        # update d
        d = -np.array(grad_f(x))
        # update fc
        fc = fc + 1
        if fc >= max_iter:
            warn('The line search algorithm did not converge', LineSearchWarning)
            alphas = None
    new_fval = f(x)
    return alphas, fc, x, new_fval

def _alpha_search_armijo(f, grad_f, x, alpha, gama, c, d):
    """
        search suitable alpha by armijo rules
    """
    x_n = x + alpha*d
    rsdl = f(x_n) - f(x) - c*alpha*np.dot(np.array(grad_f(x)), d)
    while rsdl > 1e-6:
        alpha = alpha*gama
        x_n = x + alpha*d
        rsdl = f(x_n) - f(x) - c*alpha*np.dot(np.array(grad_f(x)), d)
    return alpha
