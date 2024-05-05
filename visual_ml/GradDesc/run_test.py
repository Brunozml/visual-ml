import numpy as np

from function_examples import g, grad_g, k, grad_k

## ~~~~~~~~~~~~~~~~~~~~~ ##

## Some commands to visualize a function R^2 -> R,
## like the functions g and k that we're trying to optimize

import matplotlib.pyplot as plt

plt.close('all')                   # close the previous figures
fig = plt.figure()                 # create a new one
x0 = np.linspace(-3.0, 3.0, 1000)  # we'll be plotting in the interval [-3,3]x[-3,3]
x1 = x0
x0, x1 = np.meshgrid(x0, x1)       # this produces two grids, one with the x0 coordinates and one with the x1 coordinates
z = k(np.stack((x0, x1)))          # this computes a function (in this case g) over the stacked grids


## do a contour plot
plt.contour(x0, x1, z, 50, cmap='RdGy')
plt.savefig('results/run_test-contourplot.png')

## alterntative to contour plot: a surface plot (in 3-D, that you can rotate etc)
## to use it, comment the plt.contour line and uncomment the follwoing ones
## (note that we need some additional modules for this)

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
plt.close('all')        
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')      # prepare the figure to hold a 3-d plot
ax.plot_surface(x0, x1, z, alpha=0.5, cmap=cm.coolwarm)  # 3-d plot
plt.savefig('results/run_test-3dsurfaceplot.png')
## ~~~~~~~~~~~~~~~~~~~~~ ##

### Optimization, gradient descent version

from GradDescND import grad_desc

## we'll call the optimization variable `z` instead of `x` to avoid
## confusing it with the x coordinate
z0 = np.array([0.5, -1.0])
z_min, zs, converged = grad_desc(k, z0, max_t = 10000, alpha=0.1)

## Plot the trajectory
## When we built zs, we have accumulated a list of length-2 vectors, and then called `np.array` on
## that. This produces a matrix in which each row is one point.
## For plotting, we need all the x components first, and then all the y components:
## these are the first and second column in the matrix.
plt.close('all')        
plt.plot(zs[:,0], zs[:,1], 'x-')
plt.savefig('results/run_test-gd_trajectory0.png')
### Optimization, scipy minimize version

from scipy.optimize import minimize

## In order to plot the trajectory, we use the callback mechanism. We create a list as
## a global variable, then we use a function that, given a point, stores it in the list:
##
##   lambda xk: min_zs.append(xk)
##
## The minimize optimizer will call this function at each iteration

min_zs = [z0]
res = minimize(k, z0, callback = lambda xk: min_zs.append(xk))
min_zs = np.array(min_zs)

## plot the trajectory
plt.close('all')        
plt.plot(min_zs[:,0], min_zs[:,1], 'x-')
plt.savefig('results/run_test-gd_trajectory1.png')
