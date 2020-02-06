## Mechanics of a 5-parameter fit for RA_0, pmRA, DE_0, pmDE, and parallax
## as a function of time.


RA(t_i) = RA(t_0) + pmRA * (t_i - t_0)

obsRA(t_i) = RA(t_i) + plx * pfunc(RA(t_i), DE(t_i), X(t_i), Y(t_i), Z(t_i))
#RA_obs(t_i) = RA(t_0) + pmRA * (t_i - t_0) + plx * pfunc()

RA(t) = RA(T) + (t - T)*pmRA + plx * rpfunc(t)
DE(t) = DE(T) + (t - T)*pmDE + plx * rpfunc(t)


