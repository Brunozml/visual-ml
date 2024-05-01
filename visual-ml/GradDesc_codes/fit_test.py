"""
This script provides an example of using a non-linear minimization method
to perform a least-squares fit

Suppose that we have some non-linear function that depends on a real input x
and on some parameters a, b, c
"""
import numpy as np
import matplotlib.pyplot as plt


def s(x, a, b, c):
    return a + b * np.sin(c * x)

## Now for the sake of example we generate some data. The general idea of this
## test is:
##   1) we start with some known values of the parameters a, b, c (we call
##      these the "true" values)
##   2) we pick few values of x and get the corresponding function output,
##      using the true values of a, b, c
##   3) we add some "noise" (random jitter) to the output values
##   4) this gives us some data points (an array of x and a corresponding array
##      of y)
##   5) now we want to see if we can recover the original values of a, b, c by
##      just knowing the data points x,y and the general shape of the function s.
##      At least, we wish to get something close to the true values.

##   6) we use a "least squares" cost function to estimate how far we are from
##      the true values (more on this below)

##   7) we thus minimize the cost function using either our grad_desc algorithm
##      or scpiy.optimize.minimize
##   8) this procedure will give us some "optimal" values of a, b, c. We have a
##      look at how close we got to actually guessing the true values.
##
## This mimics the situation in which we have some data points and a model that
## we want to fit, and that model has some unknown parameters that we wish to
## infer from the data.
## In a realistic setting, this can be used to predict the result over inputs
## that we have never observed. Or, if the parameters have an interpretation
## (e.g. the "rate of production" of something), we can use the data to measure
## their value. Or both.
## NOTE: this method is general but determining whether it's appropriate and
## how to actually interpret the results depends on many things (how good is
## the model, how good is the cost function for your particular situation, etc.).
## You need a careful statistical analysis before you can make sense of the
## outcome, which is out of scope of this course.

## Ok here we go.

#%% 
## Choose some arbitrary "true" values for the parameters a, b, c
a_true, b_true, c_true = 0.5, 1.2, 3.5

# Since these parameters are what we are going to try to optimize, and
# since all our optimizers work on numpy arrays, we also store them
# in an array, for later.
# We use the letter z to denote out optimization variable, not to be
# confused with x which is the input to the function `s`.
z_true = np.array([a_true, b_true, c_true])

## Plot the true function, using a fine-grained subdivision of the x axis
## (in a realistic setting, we wouldn't have this of course)
x_fine = np.linspace(-5, 5, 1000)
y_fine = s(x_fine, a_true, b_true, c_true)
plt.clf()
plt.plot(x_fine, y_fine, '-', label="true")
plt.savefig('results/fit_test-output0.png')
## Now pick some inputs (we just take them with linrange but they could be
## random or have any other form)
x_train = np.linspace(-5, 5, 60)

## Now we evaluate the output, but add some Gaussian noise to it
np.random.seed(4627323) # make the simulation reproducible
y_train = s(x_train, a_true, b_true, c_true) + 0.1 * np.random.randn(len(x_train))

## Let's plot our data points
plt.plot(x_train, y_train, 'o', label="data")
plt.savefig('results/fit_test-output1.png')
## Suppose that, as a preliminary step, we just guess some values of the
## parameters. We can then plot the result. This works fine for visual
## inspection (which isn't always possible) but it doesn't give us a way
## to quantify how well we guessed, nor a well-defined way to improve the
## guess.
a_guess, b_guess, c_guess = 1.0, 2.0, 3.0
z_guess = np.array([a_guess, b_guess, c_guess]) # as an array, for later
y_guess = s(x_fine, a_guess, b_guess, c_guess)
# plt.plot(x_fine, y_guess, '-', label="guess") # commented-out to remove clutter

## Time to define the cost function then. This quantifies how well we are
## guessing the true parameters.
## Here we are assuming that x_data and y_data are two given arrays, and are
## fixed. They are the data points.
## What we are varying are the parameters a, b, c (we are trying to recover
## a_true, b_true, c_true).
## So we say that, for a given guess of the values a, b, c, the "cost" of an
## individual pair (x_data[i], y_data[i]) is the discrepancy between the
## observed value y_data[i] and the prediction based on the guess,
## s(x_data[i], a, b, c).
## We measure this discrepancy by taking the square of the difference (so that
## it's always positive).
## The overall discrepancy is the mean of all the individual discrepancies.
## Exploiting as usual the broadcasting rules, all this computation reduces
## to just two lines of code.
def discrepancy(a, b, c, x_data, y_data):
    ## non-broadcasting version, for reference
    # n = len(x_data)
    # s = 0.0
    # for i in range(n):
    #     v = s(x_data[i], a, b, c)
    #     s += (y_data[i] - v)**2
    # return s / n

    ## broadcasting version
    y_pred = s(x_data, a, b, c) # a vector, since x is a vector!
    return np.mean((y_data - y_pred)**2)

## Since our optimizers take an array as input, however, let's define the
## same function with an array argument instead. Also, we will use the
## name `loss` instead of `discrepancy` since that's the traditional term:
def loss(z, x_data, y_data):
    y_pred = s(x_data, z[0], z[1], z[2])
    return np.mean((y_data - y_pred)**2)

## Now that we have a measure of the discrepancy we can compute it on the
## true values (which we wouldn't be able to do in a realistic setting, but
## here it's useful to get a sense of what we should expect if everything
## works) and for the guessed values.
print(f"loss with guess: {loss(z_guess, x_train, y_train)}")
print(f"loss with true: {loss(z_true, x_train, y_train)}")

## Now we try to find the values of a, b, c that minimize the overall discrepancy.
## In other words, we try to improve on our initial guess.
## This is a 3-d optimization problem.
## On the other hand, x_train and y_train are our dataset and are considered fixed.
## So the function that we are going to optimize can be written as a lambda
## function in this way:
##
##   lambda z: loss(z, x_train, y_train)
##
## The main issue here is guessing well the initial values, and (for gradient
## descent) choosing the parameters (mainly alpha and/or max_t)

## Gradient descent version
## (NOTE: we could compute the gradient analytically...)

from GradDescND import grad_desc

## NOTE: if you have a bad initial guess, or alpha is too large, this
##       function will get stuck in a local optimum, very far from the
##       true value, and the fit will be absolute garbage.
z_opt_gd, zs_gd, converged = grad_desc(lambda zz: loss(zz, x_train, y_train),
                                       z_guess,
                                       max_t = 10000,
                                       alpha = 0.1)
print(f"loss with g.d. opt: {loss(z_opt_gd, x_train, y_train)}")

## Plot the function with the best-fit parameters
y_opt_gd = s(x_fine, z_opt_gd[0], z_opt_gd[1], z_opt_gd[2])
plt.plot(x_fine, y_opt_gd, '-', label="opt")

plt.legend()

## Scipy minimize version. Basically gives the same result as grad_desc, but in
## fewer iterations.
## The callback function is used here to collect the intermediate optimization
## values; it stores them in a global variable `zs_min` which starts out as a
## list and is eventually converted into a n array.

from scipy.optimize import minimize
zs_min = [z_guess]
res = minimize(lambda zz: loss(zz, x_train, y_train),
               z_guess,
               callback = lambda zk: zs_min.append(zk))
zs_min = np.array(zs_min)

## Assuming that it succeeded, we extract the optimal values which are
## stored inside sol.x
z_opt_min = res.x

## Since the result object from minimize includes the optimal value of the
## function (in our case, the loss), we don't even need to recompute it.
## So instead of `loss(z_opt_min, x_train, y_train)` we can just use `res.fun`
print(f"loss with min. opt: {res.fun}")

### Optimization trajectories

## Uncomment the following to see a plot of the trajectories that the
## two optimizers followed, in the 3-d space of the parameters a, b, c

# from mpl_toolkits import mplot3d
# plt.close('all')
# ax = plt.axes(projection='3d')
# ax.plot3D(zs_gd[:,0], zs_gd[:,1], zs_gd[:,2], '-x')
# ax.plot3D(zs_min[:,0], zs_min[:,1], zs_min[:,2], '-o')


### EXTRA STUFF, NOT DONE DURING THE LECTURE

## Uncomment the following to see a (partial) plot of the optimization
## landscape, showing that there are plenty of local minima that may trap you!

## We can try to get an idea of how difficult this problem really is,
## by trying to plot the optimization landscape, as we did for the
## GradDescND example functions. However, this time the problem is
## 3-dimensional, therefore we can't really plot it. What we do instead
## is fix one of the parameters to its true value, and observe how
## the discrepancy changes as a function of the remaining two variables.
## The plotting code is basically the same as in the test_run.py code,
## and you can choose whether you prefer to see a 3-d plot or a contour
## map.

# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm

# fig = plt.figure()               # create a new figure
# ax = fig.gca(projection='3d')    # prepare the figure to hold a 3-d plot
# x0 = np.linspace(-2.0, 3.0, 100) # we'll be plotting in the interval [-2,3]x[0,8]
# x1 = np.linspace(0.0, 8.0, 100)
# x0, x1 = np.meshgrid(x0, x1)     # this produces two grids of points

# gr = np.stack((x0, x1))          # stack the two grids together

## There is an extra difficulty this time in making broadcasting work, we
## can't just call f(gr) as we did in the other script. The reason is that
## we're trying to exploit broadcasting twice, once for broadcasting over
## the data points `x` and one for broadcasting over the grid of parameters
## `gr`.
## The solution is to create extra dimensions in the arrays so that the two
## broadcasting operations don't interfere with each other. This could be
## done with reshaping, here we use `np.newaxis` which is slightly nicer.
from numpy import newaxis as nx

## Note that we're fixing b=b_true and varying a and c.
# h = np.mean((s(x_train[:,nx,nx], gr[0], b_true, gr[1]) - y_train[:,nx,nx])**2, axis=0)

## If you wanted to fix a=a_true and vary b and c instead...
# h = np.sum((s(x_train[:,nx,nx], a_true, gr[0], gr[1]) - y_train[:,nx,nx])**2, axis=0)

# ax.plot_surface(x0, x1, h, alpha=0.5, cmap=cm.coolwarm)  # 3-d plot

## alternatively, do a contour plot: disable the `fig.gca` and `plot_surface`
## commands, and use this one instead
# plt.contour(x0, x1, h, 50, cmap='RdGy')

# %%
