# after running 33_joint_param_residuals.py we have arrays:
# every_rdist --> combined rdist, all 4 sensors
# every_rdelt --> combined correction, all 4 sensors

from numpy.polynomial import Chebyshev

dumb_cfit = Chebyshev.fit(every_rdist, every_rdelt, deg=5)
# Chebyshev([ 9.65739923, 12.28273255,  3.17197644,  0.01988002, -0.18699903, -0.04024   ], domain=[ 114.85589943, 2975.47913828], window=[-1.,  1.])

# Ensure coverage using manually specified domain:
use_domain = [0.0, 3100.0]
cfit = Chebyshev.fit(every_rdist, every_rdelt, deg=5, domain=use_domain)
# Chebyshev([10.1684901 , 13.30861981,  3.56958906, -0.02580189, -0.25977033, -0.06013996], domain=[   0., 3100.], window=[-1.,  1.])

# Ensure coverage using manually specified domain (higher even order ...):
use_domain = [0.0, 3100.0]
cfit6 = Chebyshev.fit(every_rdist, every_rdelt, deg=6, domain=use_domain)
# Chebyshev([10.15247764, 13.30378535,  3.54108531, -0.02906309, -0.27916618, -0.06103624, -0.00851785], domain=[   0., 3100.], window=[-1.,  1.])


