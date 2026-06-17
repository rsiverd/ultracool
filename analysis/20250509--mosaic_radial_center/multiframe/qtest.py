#!/usr/bin/env python3

def avg1(values):
    return sum(values) / len(values)

def avg2(values):
    return sum(values) / float(4)

def avg3(values):
    return np.average(values)

def avg4(values):
    return sum(values) / 4.0

# In [155]: %timeit list(avg_nstars/x for x in vals)
# 179 ns ± 1.35 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)
# 
# In [156]: %timeit [avg_nstars/x for x in vals]
# 110 ns ± 1.49 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)
# 
# In [157]: %timeit asdf=[1.0, 1.0, 1.0, 1.0]
# 13.8 ns ± 0.203 ns per loop (mean ± std. dev. of 7 runs, 100000000 loops each)
# 
# In [158]: %timeit asdf=np.ones(4, dtype='float')
# 618 ns ± 6.04 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
# 
# In [159]: %timeit ne_scale=1.0; nw_scale=1.0; se_scale=1.0; sw_scale=1.0
# 7.58 ns ± 0.0649 ns per loop (mean ± std. dev. of 7 runs, 100000000 loops each)

