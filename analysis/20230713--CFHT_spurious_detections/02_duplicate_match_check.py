#!/usr/bin/env python3
#
# Do we ever have multiple sources matching to the same Gaia star?

import os, sys, time
import numpy as np
import pandas as pd

# Load CSV data:
sys.stderr.write("Loading data file ... ")
tik = time.time()
csv_file = '20230629--final_matches.csv'
data = pd.read_csv(csv_file)
tok = time.time()
sys.stderr.write("done. Took %.3f seconds.\n" % (tok-tik))

# Group by image:
chunks = data.groupby('Image Name')

dupe_count = []
for tag,isubset in chunks:
    cbase = os.path.basename(tag)
    nsrcs = len(isubset)
    sys.stderr.write("Catalog %s has %d sources.\n" % (cbase, nsrcs))
    udecs = len(np.unique(isubset['Gaia Dec']))
    sys.stderr.write("Among these, we have %d unique Dec values.\n" % udecs)
    ndupe = nsrcs - udecs
    dupe_count.append((cbase, nsrcs, udecs, ndupe))
    if ndupe > 0:
        sys.stderr.write("--> WARNING WARNING WARNING!\n")
        sys.stderr.write("--> Have %d (possible) duplicates!\n" % ndupe)
    sys.stderr.write("\n")
    pass

# Save a summary:
save_dupes = 'dupe_count.txt'
with open(save_dupes, 'w') as sd:
    sd.write("cbase,nsrcs,udecs,ndupe\n")
    for stuff in dupe_count:
        sd.write("%s,%d,%d,%d\n" % stuff)

