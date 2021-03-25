
import astropy.time as astt

## Gaia DR2 (2015.5):
gpoch = astt.Time(2457206.375, format='jd', scale='tcb')

## Gaia DR3 (2016.0):
gpoch = astt.Time(2457389.0, format='jd', scale='tcb')

julian_year_days = 365.25
julian_half_year = 182.625

