import numpy as np
import matplotlib.pyplot as plt

def g(x):
    return x**4 + 4 * x**3 + x**2 - 10 * x + 1

def grad_g(x):
    return 4 * x**3 + 12 * x**2 + 2 * x - 10

def h(x):
    return np.log(1 + np.exp(x))

def k(x):
    return h(g(x))

xv = np.linspace(-5, 3, 1000)
yv = k(xv)

def grad(f, x, delta = 1e-5): # this is the finite-differences-based automatic derivation
    return (f(x + delta) - f(x - delta)) / (2 * delta)

plt.clf()
plt.plot(xv, yv)
plt.savefig('results/GradDesc1d_1-output.png')

# def grad_k(x):
#     return grad(k, x)

def grad_desc1d(f, x0, grad_f = None, max_t = 100, alpha = 0.01, epsilon = 1e-6):
    if grad_f is None:
        grad_f = lambda xx: grad(f, xx)

    x = x0
    # xs = np.zeros(max_t + 1)
    # xs[0] = x0
    converged = False
    xs = [x0] # we save the values we find along the way
    for k in range(max_t):
        p = grad_f(x)
        x = x - alpha * p
        # xs[k+1] = x
        xs.append(x)
        if abs(p) < epsilon:
            converged = True
            break
    xs = np.array(xs)
    return x, xs, converged
