#!/usr/bin/env python3

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

for tag,isubset in chunks:
    cbase = os.path.basename(tag)
    break

