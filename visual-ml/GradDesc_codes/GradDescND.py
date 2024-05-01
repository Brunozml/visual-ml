import numpy as np

## Generalization of the finite-differences-based automatic derivation to N dimensions.
def grad(f, x, delta = 1e-5):
    n = len(x)
    g = np.zeros(n)                         # allocate the space for the gradient
    for i in range(n):                      # cycle over the dimension
        x_old = x[i]
        x[i] = x_old + delta
        fp = f(x)                   

        x[i] = x_old - delta
        fm = f(x)                          

        x[i] = x_old

        ## compute the i-th component of the gradient and save it
        g[i] = (fp - fm) / (2 * delta)
    return g

def norm(x):
    return np.sqrt(np.sum(x**2))
    # return np.sqrt(x @ x)
    # return np.sqrt(np.dot(x, x))

## The main gradient descent function. It's nearly the same as the 1-D version.
def grad_desc(f, x0,
              grad_f = None,
              max_t = 100,
              alpha = 0.01,
              epsilon = 1e-6,
              callback = None):
    
    if grad_f is None:
        grad_f = lambda xx: grad(f, xx)

    x = x0
    xs = [x0.copy()] # we save the values we find along the way

    converged = False

    for k in range(max_t):
        p = grad_f(x)
        assert len(p) == len(x)

        x = x - alpha * p
        xs.append(x)

        # callback mechanism
        if callback is not None:
            if callback(x):
                break

        if norm(p) < epsilon:
            converged = True
            break
    
    xs = np.array(xs)

    return x, xs, converged
