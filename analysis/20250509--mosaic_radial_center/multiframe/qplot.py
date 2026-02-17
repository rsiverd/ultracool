

fig = plt.figure(1, figsize=(7, 7))
#quads = ('NE',) #, 'NW', 'SE', 'SW')
#quads = ('NW',) #, 'NW', 'SE', 'SW')
#quads = ('SE',) #, 'NW', 'SE', 'SW')
#quads = ('SW',) #, 'NW', 'SE', 'SW')
this_runid = context.runid
quads = ('NE', 'NW', 'SE', 'SW')
hkw = {'bins':100, 'range':(-2, 2), 'density':True}

## X errors:
plt.clf()
for qq in quads:
    hist(runid_xerror[qq], label=qq, **hkw)
grid()
legend()
plt.gca().set_title(this_runid)
xlabel('X error [px]')
tight_layout()
save_file = 'runid_x_error_%s.png' % this_runid
savefig(save_file)


## Y errors:
plt.clf()
for qq in quads:
    hist(runid_yerror[qq], label=qq, **hkw)
grid()
legend()
plt.gca().set_title(this_runid)
xlabel('Y error [px]')
tight_layout()
save_file = 'runid_y_error_%s.png' % this_runid
savefig(save_file)


## X,Y error vs X,Y position:
plt.clf()
qxkw = {'angles':'xy', 'scale_units':'xy', 'scale':0.1}
skw = {'lw':0, 's':5}
skw = {'lw':0, 's':3}
runid_rerror = {qq:np.hypot(runid_xerror[qq], runid_yerror[qq]) for qq in runid_xerror.keys()}
runid_rpixel = {qq:np.hypot(runid_xpixel[qq], runid_xpixel[qq]) for qq in runid_xpixel.keys()}
#every_rerror = np.concatenate(list(runid_rerror.values()))
#skw = {'lw':0} #, 's':25}
fig, axs = plt.subplots(2, 2, num=1, clear=True)
axs[0,0].scatter(runid_xpixel['NE'], runid_ypixel['NE'], 
                c=10*runid_rerror['NE'], **skw)
axs[0,1].scatter(runid_xpixel['NW'], runid_ypixel['NW'], 
                c=10*runid_rerror['NW'], **skw)
axs[1,0].scatter(runid_xpixel['SE'], runid_ypixel['SE'], 
                c=10*runid_rerror['SE'], **skw)
axs[1,1].scatter(runid_xpixel['SW'], runid_ypixel['SW'], 
                c=10*runid_rerror['SW'], **skw)
axs[0,0].set_title(this_runid)
#axs[0,0].quiver(runid_xpixel['NE'], runid_ypixel['NE'], 
#                runid_xerror['NE'], runid_yerror['NE'], **qxkw)
tight_layout()
save_file = 'runid_qq_r_error_%s.png' % this_runid
savefig(save_file)


## Radial error vs radial position (breakout)
plt.clf()
qxkw = {'angles':'xy', 'scale_units':'xy', 'scale':0.1}
skw = {'lw':0, 's':5}
skw = {'lw':0, 's':1}
runid_rerror = {qq:np.hypot(runid_xerror[qq], runid_yerror[qq]) for qq in runid_xerror.keys()}
runid_rpixel = {qq:np.hypot(runid_xpixel[qq], runid_xpixel[qq]) for qq in runid_xpixel.keys()}
#every_rerror = np.concatenate(list(runid_rerror.values()))
#skw = {'lw':0} #, 's':25}
fig, axs = plt.subplots(2, 2, num=1, clear=True)
axs[0,0].scatter(runid_rpixel['NE'], runid_rerror['NE'], **skw)
axs[0,1].scatter(runid_rpixel['NW'], runid_rerror['NW'], **skw)
axs[1,0].scatter(runid_rpixel['SE'], runid_rerror['SE'], **skw)
axs[1,1].scatter(runid_rpixel['SW'], runid_rerror['SW'], **skw)
axs[0,0].set_title(this_runid)
#axs[0,0].quiver(runid_xpixel['NE'], runid_ypixel['NE'], 
#                runid_xerror['NE'], runid_yerror['NE'], **qxkw)
tight_layout()
save_file = 'runid_qq_rpix_err_%s.png' % this_runid
savefig(save_file)

