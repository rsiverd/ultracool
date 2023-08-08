
from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris, EarthLocation
from astropy.coordinates import get_body_barycentric, get_body

# On first use, this will download the DE430 ephemeris:
solar_system_ephemeris.set('de430')

# You can specify an observatory site as EarthLocation. To see a list
# of known observatories, try:
sites = EarthLocation.get_site_names()

cfht_observatory = EarthLocation.of_site('cfht')


salt_observatory = EarthLocation.of_site('salt')

