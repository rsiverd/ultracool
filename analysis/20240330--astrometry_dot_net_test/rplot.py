
def get_pscale(cdm):
    this_cdm = cdm.reshape(2,2)
    cd_pscales = np.sqrt(np.sum(this_cdm**2, axis=1))    # sum along rows
    norm_cdmat = this_cdm / cd_pscales                         # this should yield a rotation matrix
    cd_ang_rad = np.arccos(norm_cdmat[0,0])                  # arccos of first element should work
    cd_ang_deg = np.degrees(cd_ang_rad)                      # now in degrees
    return cd_pscales,cd_ang_deg


cdm_vals = [3600.*big_results[x]['x'][:4] for x in use_runids]

pscales, pos_angs = zip(*[get_pscale(x) for x in cdm_vals])

fig = plt.figure(1)
fig.clf()

ax = fig.add_subplot(111)
ax.grid(True)

std_devs = np.array([jresid_stddev[x] for x in use_runids])*3600.0
mad_devs = np.array([jresid_maddev[x] for x in use_runids])*3600.0
dummy_id = np.arange(len(std_devs))

skw = {'lw':0, 's':25}
ax.scatter(dummy_id, std_devs, label='STD', **skw)
ax.scatter(dummy_id, mad_devs, label='MAD', **skw)
ax.set_ylabel('RMS error [arcsec]')
ax.set_xlabel("RUNID")
ax.set_ylim(0, 0.35)
ax.legend(loc='upper left')
plt.xticks(dummy_id, use_runids)
for label in ax.get_xticklabels():
       label.set_rotation(90)
fig.tight_layout()
fig.savefig('runid_residuals.png', bbox_inches='tight')

