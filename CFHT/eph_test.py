
from astroquery.jplhorizons import Horizons

cfh_kw = {     'body':   399,           # Earth
                'lon':  -155.47166667,
                'lat':   +19.82829444,
          'elevation':     4.2094}

ssb_loc     = '@0'
cfh_loc     = 'T14'
earth_bary  = '399'
refplane    = 'earth'
dummy_epoch = 2454483.84247

obj = Horizons(id=cfh_kw, location=ssb_loc, epochs=dummy_epoch)




# no workie:
obj = Horizons(id=cfh_loc, location=ssb_loc, epochs=dummy_epoch)

-----------------------------------------------------------------------
-----------------------------------------------------------------------
-----------------------------------------------------------------------

compare_cols = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'lighttime', 'range', 'range_rate']

# this works, but might be negative of desired coordinates:
cfh_obj = Horizons(id=ssb_loc, location=cfh_loc, epochs=dummy_epoch)
cfh_vecs = cfh_obj.vectors(refplane=refplane)
print(cfh_vecs)
#          targetname          datetime_jd  ...       range             range_rate     
#             ---                   d       ...         AU                AU / d       
# --------------------------- ------------- ... ----------------- ---------------------
# Solar System Barycenter (0) 2454483.84247 ... 0.988186142137829 0.0001637850587503002
cfh_xyz = [cfh_vecs[k] for k in ( 'x',  'y',  'z')]
cfh_vel = [cfh_vecs[k] for k in ('vx', 'vy', 'vz')]
cfh_chk = [cfh_vecs[k].value for k in compare_cols]
sys.stderr.write("posn  X, Y, Z: %.4f, %.4f, %.4f\n" % tuple(cfh_xyz))
sys.stderr.write("vels VX,VY,VZ: %.4f, %.4f, %.4f\n" % tuple(cfh_vel))
# vecs X,Y,Z: 0.4541, -0.8053, -0.3490

-----------------------------------------------------------------------
-----------------------------------------------------------------------
-----------------------------------------------------------------------

# compare to Earth vectors:
earth_obj = Horizons(id='399', location=ssb_loc, epochs=dummy_epoch)
earth_vecs = earth_obj.vectors(refplane=refplane)
print(earth_vecs)
#  targetname  datetime_jd           datetime_str          ...       range              range_rate     
#     ---           d                    ---               ...         AU                 AU / d       
# ----------- ------------- ------------------------------ ... ------------------ ---------------------
# Earth (399) 2454483.84247 A.D. 2008-Jan-18 08:13:09.4080 ... 0.9881502458094619 2.887716144468134e-05
earth_xyz = [earth_vecs[k] for k in ( 'x',  'y',  'z')]
earth_vel = [earth_vecs[k] for k in ('vx', 'vy', 'vz')]
earth_chk = [earth_vecs[k].value for k in compare_cols]
sys.stderr.write("posn  X, Y, Z: %.4f, %.4f, %.4f\n" % tuple(earth_xyz))
sys.stderr.write("vels VX,VY,VZ: %.4f, %.4f, %.4f\n" % tuple(earth_vel))
# vecs X,Y,Z: -0.4541, 0.8052, 0.3490


-----------------------------------------------------------------------
-----------------------------------------------------------------------
-----------------------------------------------------------------------


# -----------------------------------------------------------------------
# statue of liberty example (ONLY WORKS FOR ephemerides() METHOD!):
statue_of_liberty = {'lon': -74.0466891, 'lat': 40.6892534, 'elevation': 0.093}
sol_obj = Horizons(id='Ceres', location=statue_of_liberty, epochs=2458133.33546)
print(sol_obj)
sol_vecs = sol_obj.vecs()

# -----------------------------------------------------------------------
# topocentric Chang'e-2 example:
ce_2 = {'lon': 23.522, 'lat': 0.637, 'elevation': 181.2, 'body': 301}
double = {'lon': 23.47, 'lat': 0.67, 'elevation': 0, 'body': 301}
obj = Horizons(id=double, location=ce_2, epochs=2454483.84247)
vecs = obj.vectors()
distance_km = (vecs['x'] ** 2 + vecs['y'] ** 2 + vecs['z'] ** 2) ** 0.5 * 1.496e8
print(f"{distance_km.value.data[0]:.3f}")
# this raises an error:
ValueError: cannot use topographic position in statevectors query


-----------------------------------------------------------------------
-----------------------------------------------------------------------
-----------------------------------------------------------------------

compare_cols = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'lighttime', 'range', 'range_rate']

# Do Earth <--> SSB check in forward and reverse ... compare
essb_obj = Horizons(id=ssb_loc, location=earth_bary, epochs=dummy_epoch)
ssbe_obj = Horizons(id=earth_bary, location=ssb_loc, epochs=dummy_epoch)

essb_vec = essb_obj.vectors(refplane=refplane)
ssbe_vec = ssbe_obj.vectors(refplane=refplane)

essb_chk = np.array([essb_vec[k].value for k in compare_cols])
ssbe_chk = np.array([ssbe_vec[k].value for k in compare_cols])

