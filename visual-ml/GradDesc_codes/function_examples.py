import numpy as np

## Some function examples, for testing purposes

## A function from R^2 to R. It's a paraboloid, so that it's convex (it has a
## single global minimum), and moreover such that finding the minimum
## analytically is actually easy and we can check our results
## (the minimum is at x=(-11/6, -1/3))
def g(x):
    return x[0]**2 + 4 * x[1]**2 - 2 * x[0] * x[1] + 3 * x[0] - x[1] - 1

## We can compute the gradient of g explicitly
## Since the domain of g is R^2, the gradient will have 2 components
def grad_g(x):
    dg_0 = 2 * x[0] - 2 * x[1] + 3
    dg_1 = 8 * x[1] - 2 * x[0] - 1
    return np.array([dg_0, dg_1])

## This is a function from R^2 to R^2. We cannot optimize this
## directly (it's not well-defined what it means to seek a minimum of h,
## since h outputs 2 numbers...)
def h(x):
    h0 = np.exp(0.01 * x[0]**2 - 0.005 * x[0] * (x[1] - 0.1))
    h1 = np.exp(0.01 * x[1]**2 - 0.005 * x[1] + 0.3)
    return np.array([h0, h1])

## The "gradient" of g is actually a matrix, called the Jacobian.
## We just need to compute the gradient of each component separately,
## and stack them all togehter.
def grad_h(x):
    ## If you look at the structure of h, it's clear that in this case it's
    ## useful to first compute the values of h(x)
    h0, h1 = h(x)
    ## Here, dhi_j is the derivative of the i-th component of h with respect to
    ## the j-th component of x
    dh0_0 = h0 * (0.02 * x[0] - 0.005 * (x[1] - 0.1))
    dh0_1 = h0 * (-0.005 * x[0])
    dh1_0 = h1 * 0.0 # the h1 is here for broadcasting purposes if x is a matrix or higher-dimensional array
    dh1_1 = h1 * (0.02 * x[1] - 0.005)
    # return np.array([dh0, dh1])
    return np.array([[dh0_0, dh0_1], [dh1_0, dh1_1]])

## This is a function from R^2 to R. It't a composition of h and g. It's still
## convex, but the minimum for this function cannot be found analytically.
## It's also useful to test the chain rule when computing derivatives in
## arbitrary dimensions
def k(x):
    return g(h(x))

## The gradient of k can be computed using the chain rule. It works in the same
## way as the standard chain rule that you use in the R -> R case, except that
##   1) derivatives become Jacobians (i.e. gradients if the output is 1-d)
##   2) multiplication is intended as the matrix multiplication
##   3) it may be necessary to transpose the Jacobian matrix, depending on
##      the convention used to write things
##
## In our case, we have g(h(x)) where x is in R^2 and h: R^2 -> R^2 and
## g: R^2 -> R
##
## Let's write y = h(x), and then y is the input to g.
##
## So the derivative of k with respect to the i-th component of x is:
##
##   dk/dx[i] = sum(dg/dy[0] * dh[0]/dx[i] + dg/dy[1] * dh[1]/dx[i])
##
## which is the (matrix) product between the i-th column of the Jacobian of h
## with the gradient of g.
##
## So overall, the gradient dk/dx is the (matrix) product of the Jacobian of h
## (transposed) with the gradient of g.

def grad_k(x):
    ## NOTE: we need the matrix product! If we try regular multiplicatioin, it
    ##       will try to broadcast and return a matrix instead of a vector!
    ##       In numpy, matrix multiplication can be performed in several ways,
    ##       the easiest one is to use `@` instead of `*`. Alternatively, one
    ##       can use np.dot or np.matmul
    gg = grad_g(h(x))
    gh = grad_h(x)
    return gh.T @ gg         # the `@` means matrix multiplication
    #return np.dot(gh.T, gg) # alternative expression
