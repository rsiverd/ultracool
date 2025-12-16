
badrun = '12BQ03'

# image numbers poly=1, NE:
shifts[1]['NE']['ii']
rpick1 = (shifts[1]['NE']['rr'] == badrun)
shifts[1]['NE']['ii'][rpick1]

# image numbers poly=2, NE:
shifts[2]['NE']['ii']
rpick2 = (shifts[2]['NE']['rr'] == badrun)
shifts[2]['NE']['ii'][rpick2]

# do they match?
shifts[1]['NE']['ii'][rpick1] == shifts[2]['NE']['ii'][rpick2]

badinum = shifts[1]['NE']['ii'][rpick1]

lookie = all_data[all_data.inum.isin(badinum)]

# ----------------------------------------------------------------------- 

# poly=1, NE, RUNIDs
shifts[1]['NE']['rr']

# per-runid data point tally:
rtally = {x:np.sum(shifts[1]['NE']['rr'] == x) for x in every_runid}

# ----------------------------------------------------------------------- 

asdf = all_data[all_data.runid == '13BQ06']
yippee = asdf.groupby(['inum', 'quad'])

easy = all_data.groupby(['runid', 'quad', 'inum'])
for blabla,subset in easy:
    if len(subset) != 2:
        sys.stderr.write("bark bark bark!\n")
        break


