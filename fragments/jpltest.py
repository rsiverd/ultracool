# Potentially relevant URLs:
#
# https://keflavich-astropy.readthedocs.io/en/latest/api/astropy.coordinates.EarthLocation.html
# https://astropy-astrofrog.readthedocs.io/en/latest/coordinates/
# https://docs.astropy.org/en/stable/api/astropy.coordinates.get_body_barycentric.html#astropy.coordinates.get_body_barycentric
# https://docs.astropy.org/en/stable/api/astropy.coordinates.get_body_barycentric_posvel.html#astropy.coordinates.get_body_barycentric_posvel
# https://docs.astropy.org/en/stable/generated/examples/coordinates/plot_obs-planning.html
# https://docs.astropy.org/en/stable/coordinates/index.html#module-astropy.coordinates
# https://docs.astropy.org/en/stable/_modules/astropy/coordinates/solar_system.html
# https://docs.astropy.org/en/stable/coordinates/transforming.html

import numpy as np
import matplotlib.pyplot as plt

from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris, EarthLocation
from astropy.coordinates import get_body_barycentric, get_body
from astropy.coordinates import get_body_barycentric_posvel
from astropy.coordinates import GCRS, ICRS

# On first use, this will download the DE430 ephemeris:
solar_system_ephemeris.set('de430')

# You can specify an observatory site as EarthLocation. To see a list
# of known observatories, try:
sites = EarthLocation.get_site_names()

cfht_observatory = EarthLocation.of_site('cfht')


salt_observatory = EarthLocation.of_site('salt')

jan1_2005 = 2453371.5
jan1_2026 = 2461041.5

date_range = Time(np.linspace(jan1_2005, jan1_2026), scale='utc', format='jd')

date_range = Time(np.linspace(jan1_2005, jan1_2005 + 0.9), scale='utc', format='jd')

#get_body_barycentric('

earth_position = get_body_barycentric('earth', date_range)

cfht_in_gcrs = cfht_observatory.get_gcrs(date_range)
cfht_gcrs_pos, cfht_gcrs_vel = cfht_observatory.get_gcrs_posvel(date_range)

cfht_in_icrs = cfht_in_gcrs.transform_to(ICRS())

skw = {'lw':0, 's':3}
fig = plt.figure(1, figsize=(12,3))
fig.clf()
ax1 = fig.add_subplot(111, aspect='equal')
#ax1.plot(cfht_gcrs_pos.x, cfht_gcrs_pos.y)
#ax1.scatter(cfht_in_icrs.cartesian.x, cfht_in_icrs.cartesian.y, **skw)
ax1.plot(earth_position.x.to('km'), earth_position.y.to('km'), label='geocenter')
ax1.plot(cfht_in_icrs.cartesian.x.to('km'), cfht_in_icrs.cartesian.y.to('km'), label='CFHT')
ax1.legend(loc='upper left')
fig.tight_layout()
plt.show()
fig.savefig('cfht_vs_earth.png')

