# Let's look at the relative speeds of CD matrix creation with different
# operations.
#
import os, sys, time
import numpy as np
import math

## Reflection and identity matrices:
xref_mat = np.array([[1.0, 0.0], [0.0, -1.0]])
yref_mat = np.array([[-1.0, 0.0], [0.0, 1.0]])
xflip_mat = yref_mat
yflip_mat = xref_mat
ident_mat = np.array([[1.0, 0.0], [0.0, 1.0]])

## -----------------------------------------------------------------------

## Possibly-speedier rotation matrix:
def fastrot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c, -s), (s, c)))

def fastrot_nocomma(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array(((c, -s), (s, c)))

def veryfastrot(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array(((c, -s), (s, c)))

def veryfastrot_nocomma(theta):
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array(((c, -s), (s, c)))

def superfastrot(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array((c, -s, s, c)).reshape(2, 2)

## Tuple-based version of rotation_matrix:
def tup_rotation_matrix(theta):
    """Generate 2x2 rotation matrix for specified input angle (radians)."""
    return np.array(((np.cos(theta), -np.sin(theta)),
                        (np.sin(theta), np.cos(theta))))

## -----------------------------------------------------------------------

## Rotation matrix builder from tangent_proj:
def rotation_matrix(theta):
    """Generate 2x2 rotation matrix for specified input angle (radians)."""
    return np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])

## The function from tangent_proj:
def make_cdmat(pa_deg, pscale):
    pa_rad = np.radians(pa_deg)
    thisCD = np.matmul(xflip_mat, rotation_matrix(pa_rad)) * (pscale / 3600.)
    return thisCD

## A faster version:
def fast_make_cdmat(pa_deg, pscale):
    pa_rad = math.radians(pa_deg)
    thisCD = np.matmul(xflip_mat, superfastrot(pa_rad)) * (pscale / 3600.)
    return thisCD

# %timeit rotation_matrix(0.002)
# 4.19 µs ± 72.6 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)


# Tuple-based rotation matrix:
# %timeit tup_rotation_matrix(0.002)
# 4.04 µs ± 72.6 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
# 4.06 µs ± 74.1 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

# Faster with fewer sin/cos?
# %timeit fastrot(0.002)
# 2.65 µs ± 22.1 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
# 2.61 µs ± 12.9 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
# 2.67 µs ± 53.2 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

# Faster with no comma? Answer is NO:
# %timeit fastrot_nocomma(0.002)
# 2.65 µs ± 35 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
# 2.66 µs ± 17.3 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
# 2.64 µs ± 28 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

# Faster with math instead of numpy for single values?
# %timeit veryfastrot(0.002)
# 1.68 µs ± 6.71 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)

# Any difference with nocomma?  None at all.
# %timeit veryfastrot_nocomma(0.002)
# 1.69 µs ± 3.27 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)

# Finally, create array flat and reshape:
# %timeit superfastrot(0.002)
# 1.23 µs ± 10.8 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
# 1.23 µs ± 8.15 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
# 1.23 µs ± 10.5 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)

cdm_1 = make_cdmat(1.0, 0.306)
# array([[-0.00008499,  0.00000148],
#        [ 0.00000148,  0.00008499]])

np.matmul(xflip_mat, rotation_matrix(0.001))
# array([[-0.9999995,  0.001    ],
#        [ 0.001    ,  0.9999995]])

np.dot(xflip_mat, rotation_matrix(0.001))
# array([[-0.9999995,  0.001    ],
#        [ 0.001    ,  0.9999995]])

# Do matmul and dot work the same?
mul_result = np.matmul(xflip_mat, rotation_matrix(0.001))
dot_result = np.dot(xflip_mat, rotation_matrix(0.001))
same = np.all(mul_result == dot_result)
sys.stderr.write("Identical: %s\n" % same)

# Do they take the same length of time?
# %timeit mul_result = np.matmul(xflip_mat, rotation_matrix(0.001))
# 6.59 µs ± 51.7 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
# 6.52 µs ± 8.52 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
# 6.52 µs ± 26.1 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
# %timeit dot_result = np.dot(xflip_mat, rotation_matrix(0.001))
# 6.97 µs ± 46.6 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
# 6.96 µs ± 16.3 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
# 7.05 µs ± 34.4 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

#%timeit np.dot(xflip_mat, rotation_matrix(0.001), out=dot_result)

# The nominal CD matrix maker:
# %timeit make_cdmat(1.0, 0.306)
# 10.5 µs ± 47.4 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
# 10 µs ± 21 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
# 9.79 µs ± 31.8 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)


# A faster version:
# %timeit fast_make_cdmat(1.0, 0.306)
# 4.73 µs ± 41.9 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
# 4.59 µs ± 28.4 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
# 4.67 µs ± 55.2 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

