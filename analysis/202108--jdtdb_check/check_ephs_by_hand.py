
## Use these commands to inspect the results from 01_get_SST_ephemeris.py
## NOTE: set:
## fhe._debug = True

plt.figure()

clf()
grid()

batches = fhe._use_batches
unstack = fhe._not_stacked
firstbad = unstack[1]

for ww,rr in zip(batches, unstack):
    diffs = rr['datetime_jd'] - ww
    sys.stderr.write("diffs: %s\n" % str(diffs))



recd_jd = np.array(fhe._raw_result['datetime_jd'])
want_jd = timestamps.tdb.jd
jdstart = want_jd[0]
jdindex = np.arange(len(want_jd))

qmax = 50
nchunks = jdindex.size // qmax + 1

ibatches = np.array_split(jdindex, nchunks)
ilastdex = [bb[-1] for bb in ibatches]


#show_recd = recd_jd - jdstart
#show_want = want_jd - jdstart

jd_diff = recd_jd - want_jd

scatter(jdindex, jd_diff, lw=0, s=9)
ylabel("JD Received - JD Expected (days)")
xlabel("JD index")

for ll in ilastdex:
    axvline(ll+0.5, ls=':', c='r')






