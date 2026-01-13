# try to find best-fit CRVAL1/2 given the input CRPIX1/2 ....

import scipy.optimize as opti


ny, nx = 11, 11
x_list = (0.5 + np.arange(nx)) / nx - 0.5            # relative (centered)
y_list = (0.5 + np.arange(ny)) / ny - 0.5            # relative (centered)
degstep = 1.0 / 3600. * 2.0

guess_posang = -0.032
guess_crpix1, guess_crpix2 = 2122.691, -81.679
#guess_crpix1 -= 1.0
#guess_crpix2 -= 1.0
#guess_crpix1 += 1.0
#guess_crpix2 -= 1.0
guess_crval1, guess_crval2 = 294.590165, 35.118152
cos_dec = np.cos(np.radians(guess_crval2))
xx, yy = np.meshgrid(x_list * degstep * cos_dec, y_list * degstep)
#test_crpix1 += 2.0
#test_crpix2 -= 2.0
sensor_crpix = sg.get_4sensor_crpix(guess_crpix1, guess_crpix2)
pctcheck = [5, 95]
cdm_calc = helpers.make_four_cdmats(test_posang)
#cdm_calc = helpers.make_four_cdmats(+0.032)
use_cdm_vals = cdm_calc
best_badness = 1e15
best_crvals = ()
for i,(dx, dy) in enumerate(zip(xx.ravel(), yy.ravel())):
    test_crval1 = guess_crval1 + dx
    test_crval2 = guess_crval2 + dy
    sys.stderr.write("Test CRVALx: %.4f, %.4f\n" % (test_crval1, test_crval2))
    rrel, rerr, terr, nrxerr, nryerr = [], [], [], [], []
    for qq,gst in gstars.items():
        tcpx1, tcpx2 = sensor_crpix.get(qq)
        gxx, gyy = gst['XWIN_IMAGE'], gst['YWIN_IMAGE']
        cdmcrv = np.array(use_cdm_vals.get(qq).tolist() + [test_crval1, test_crval2])
        test_xrel, test_yrel = helpers.inverse_tan_cdmcrv(cdmcrv,
                                    gstars[qq]['gra'], gstars[qq]['gde'])
        test_xccd = test_xrel + tcpx1
        test_yccd = test_yrel + tcpx2
        x_error = test_xccd - gxx.values
        y_error = test_yccd - gyy.values
        # error percentiles:
        #xerrpct = np.percentile(x_error, pctcheck)
        #yerrpct = np.percentile(y_error, pctcheck)
        #sys.stderr.write("%s X 5th/95th: %7.3f, %7.3f\n" % (qq, *xerrpct))
        #sys.stderr.write("%s Y 5th/95th: %7.3f, %7.3f\n" % (qq, *yerrpct))
    
        # unit vector towards CRPIX:
        test_xy_rel = np.array((test_xrel, test_yrel))
        _vec_length = np.sqrt(np.sum(test_xy_rel**2, axis=0))
        _src_v_unit = test_xy_rel / _vec_length
        _src_v_errs = np.array((x_error, y_error))
        #errdot = x_error*_src_v_unit[0] + y_error*_src_v_unit[1]
        # magnitude of error radial component:
        radial_emag = x_error*_src_v_unit[0] + y_error*_src_v_unit[1]
        # error radial component as vector:
        radial_evec = _src_v_unit * radial_emag
        nonrad_evec = _src_v_errs - radial_evec
        rrel.extend(np.hypot(test_xrel, test_yrel).tolist())
        rerr.extend(np.sqrt(np.sum(radial_evec**2, axis=0)).tolist())
        terr.extend(np.sqrt(np.sum(nonrad_evec**2, axis=0)).tolist())
        nrxerr.append(nonrad_evec[0])
        nryerr.append(nonrad_evec[1])
    nrxerr = np.concatenate(nrxerr)
    nryerr = np.concatenate(nryerr)
    nr_bad = np.sum(nrxerr**2) + np.sum(nryerr**2)
    sys.stderr.write("Total badness: %.3f\n\n" % nr_bad)
    if nr_bad < best_badness:
        best_badness = nr_bad
        best_crvals = (test_crval1, test_crval2)

def crv_badness(crvals):
    test_crval1, test_crval2 = crvals
    rrel, nrerr, nrxerr, nryerr = [], [], [], []
    for qq,gst in gstars.items():
        tcpx1, tcpx2 = sensor_crpix.get(qq)
        gxx, gyy = gst['XWIN_IMAGE'], gst['YWIN_IMAGE']
        cdmcrv = np.array(use_cdm_vals.get(qq).tolist() + [test_crval1, test_crval2])
        test_xrel, test_yrel = helpers.inverse_tan_cdmcrv(cdmcrv,
                                    gstars[qq]['gra'], gstars[qq]['gde'])
        test_xccd = test_xrel + tcpx1
        test_yccd = test_yrel + tcpx2
        x_error = test_xccd - gxx.values
        y_error = test_yccd - gyy.values
        # unit vector towards CRPIX:
        test_xy_rel = np.array((test_xrel, test_yrel))
        _vec_length = np.sqrt(np.sum(test_xy_rel**2, axis=0))
        _src_v_unit = test_xy_rel / _vec_length
        _src_v_errs = np.array((x_error, y_error))
        # magnitude of error radial component:
        radial_emag = x_error*_src_v_unit[0] + y_error*_src_v_unit[1]
        # error radial component as vector:
        radial_evec = _src_v_unit * radial_emag
        nonrad_evec = _src_v_errs - radial_evec
        rrel.extend(np.hypot(test_xrel, test_yrel).tolist())
        #rerr.extend(np.sqrt(np.sum(radial_evec**2, axis=0)).tolist())
        nrerr.append(nonrad_evec.ravel())
        #nrxerr.append(nonrad_evec[0])
        #nryerr.append(nonrad_evec[1])
    #nrxerr = np.concatenate(nrxerr)
    #nryerr = np.concatenate(nryerr)
    every_nonrad_err = np.concatenate(nrerr)
    #nr_bad = np.sum(nrxerr**2) + np.sum(nryerr**2)
    #altbad = np.sum(nrterr**2)
    return every_nonrad_err**2
    #return np.sum(every_nonrad_err**2)

crv_guess = np.array([294.590165, 35.118152])
answer = opti.least_squares(crv_badness, crv_guess)
sys.stderr.write("guess_crpix1, guess_crpix2 = %.3f, %.3f\n"
        % (guess_crpix1, guess_crpix2))

# ----------------------------------------------------------------------- 

test_r2_coeff = 0.00000254
# params = [posang, crpix1, crpix2, crval1, crval2]
def multi_badness(params):
    test_posang, \
            test_crpix1, test_crpix2, \
            test_crval1, test_crval2 = params
    sensor_crpix = sg.get_4sensor_crpix(test_crpix1, test_crpix2)
    
    use_cdm_vals = helpers.make_four_cdmats(test_posang)
    #cdm_calc = helpers.make_four_cdmats(+0.032)

    #test_crval1, test_crval2 = crvals
    #rrel, nrerr, nrxerr, nryerr = [], [], [], []
    resid = []
    for qq,gst in gstars.items():
        #if qq == 'SW':
        #    continue
        tcpx1, tcpx2 = sensor_crpix.get(qq)
        gxx, gyy = gst['XWIN_IMAGE'], gst['YWIN_IMAGE']
        cdmcrv = np.array(use_cdm_vals.get(qq).tolist() + [test_crval1, test_crval2])
        test_xrel, test_yrel = helpers.inverse_tan_cdmcrv(cdmcrv,
                                    gstars[qq]['gra'], gstars[qq]['gde'])
        test_xccd = test_xrel + tcpx1
        test_yccd = test_yrel + tcpx2
        x_error = test_xccd - gxx.values
        y_error = test_yccd - gyy.values
        # unit vector towards CRPIX:
        test_xy_rel = np.array((test_xrel, test_yrel))
        _vec_length = np.sqrt(np.sum(test_xy_rel**2, axis=0))
        _src_v_unit = test_xy_rel / _vec_length
        _src_v_errs = np.array((x_error, y_error))
        # radial distance (towards CRPIX):
        _src_R_dist = np.hypot(test_xrel, test_yrel)
        # radial distortion correction:
        _src_R_corr = test_r2_coeff * _src_R_dist**2
        _src_v_corr = _src_R_corr * _src_v_unit

        # distortion-corrected errors (residuals):
        resid_xy_vec = _src_v_errs - _src_v_corr

        resid.append(resid_xy_vec.ravel())

    every_resid = np.concatenate(resid)
    #altbad = np.sum(nrterr**2)
    return every_resid**2

multi_guess = np.array([-0.032, 2122.691, -81.679, 294.590165, 35.118152])
answer = opti.least_squares(multi_badness, multi_guess)
#sys.stderr.write("guess_crpix1, guess_crpix2 = %.3f, %.3f\n"
#        % (guess_crpix1, guess_crpix2))


# ----------------------------------------------------------------------- 

#test_r2_coeff = 0.00000254
#def crvd_badness(crvals):
#    test_crval1, test_crval2 = crvals
#    #rrel, nrerr, nrxerr, nryerr = [], [], [], []
#    resid = []
#    for qq,gst in gstars.items():
#        tcpx1, tcpx2 = sensor_crpix.get(qq)
#        gxx, gyy = gst['XWIN_IMAGE'], gst['YWIN_IMAGE']
#        cdmcrv = np.array(use_cdm_vals.get(qq).tolist() + [test_crval1, test_crval2])
#        test_xrel, test_yrel = helpers.inverse_tan_cdmcrv(cdmcrv,
#                                    gstars[qq]['gra'], gstars[qq]['gde'])
#        test_xccd = test_xrel + tcpx1
#        test_yccd = test_yrel + tcpx2
#        x_error = test_xccd - gxx.values
#        y_error = test_yccd - gyy.values
#        # unit vector towards CRPIX:
#        test_xy_rel = np.array((test_xrel, test_yrel))
#        _vec_length = np.sqrt(np.sum(test_xy_rel**2, axis=0))
#        _src_v_unit = test_xy_rel / _vec_length
#        _src_v_errs = np.array((x_error, y_error))
#        # radial distance (towards CRPIX):
#        _src_R_dist = np.hypot(test_xrel, test_yrel)
#        # radial distortion correction:
#        _src_R_corr = test_r2_coeff * _src_R_dist**2
#        _src_v_corr = _src_R_corr * _src_v_unit
#
#        # distortion-corrected errors (residuals):
#        resid_xy_vec = _src_v_errs - _src_v_corr
#
#        resid.append(resid_xy_vec.ravel())
#
#    every_resid = np.concatenate(resid)
#    #altbad = np.sum(nrterr**2)
#    return every_resid**2
#
#crvd_guess = np.array([294.590165, 35.118152])
#answer = opti.least_squares(crvd_badness, crvd_guess)
#sys.stderr.write("guess_crpix1, guess_crpix2 = %.3f, %.3f\n"
#        % (guess_crpix1, guess_crpix2))

