#!/usr/bin/env python3

import os, sys, time
import numpy as np
import pandas as pd

# Load example images (make one region file for each):
bads_list = 'examples.txt'
with open(bads_list, 'r') as bl:
    check_imgs = [x.strip() for x in bl.readlines()]

# Load CSV data:
sys.stderr.write("Loading data file ... ")
tik = time.time()
csv_file = '20230629--final_matches.csv'
data = pd.read_csv(csv_file)
tok = time.time()
sys.stderr.write("done. Took %.3f seconds.\n" % (tok-tik))

# Region file maker:
def make_regfile(rpath, detections, r1=2.0, r2=4.0):
    with open(rpath, 'w') as rf:
        for tx,ty in zip(detections['X Pixel'], detections['Y Pixel']):
            rf.write("image; annulus(%.4f, %.4f, %.1f, %.1f) #color=red\n"
                    % (tx, ty, r1, r2))
    return

# Where to save things:
save_dir = 'regions'
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

# Iterate over images, make subsets:
for image in check_imgs:
    hits = (data['Image Name'].str.find(image) >= 0)
    subset = data[hits]
    rsave = '%s/%s.reg' % (save_dir, image)
    make_regfile(rsave, subset)

