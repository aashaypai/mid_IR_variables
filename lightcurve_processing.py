import numpy as np

from astropy.io import fits


from astropy.table import Table
from astropy.time import Time
from astropy.timeseries import TimeSeries, aggregate_downsample
import astropy.units as u

from scipy.optimize import minimize

from sklearn.gaussian_process import kernels, GaussianProcessRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
###################################################################################################################################################

def nsigma_clipping(ts, n):
    """perform N-sigma clipping on timeseries to remove points that lie outside N standard deviations from the median

    Parameters
    ----------

    ts : astropy.Timeseries Object

    Returns
    -------

    ts : astropy.Timeseries Object

    """
    mask = ((ts['mag'] < np.nanmedian(ts['mag'])+n*np.std(ts['mag'])) & (ts['mag'] > np.nanmedian(ts['mag'])-n*np.std(ts['mag'])))

    ts = ts[mask]

    return ts



def local_nsigma_clipping(ts, n, size=5, verbose=False):
    """perform n-sigma clipping on points using the median of local values to remove points that lie outside n standard deviations from the median

    Parameters
    ----------

    ts : astropy.Timeseries Object

    n : np.float64
        number of standard deviations to clip outside of
    
    size : np.int32
        number of points to consider to calculate local median 

    Returns
    -------

    ts_new : astropy.Timeseries Object

    """
    clipped_count = 0

    time = ts['time'].to_value('decimalyear')
    mag = ts['mag']
    mag_err = ts['mag_err']

    time_new, mag_new, mag_err_new = np.array([]), np.array([]), np.array([])

    time_new = np.append(time_new, time[:size])
    mag_new = np.append(mag_new, mag[:size])
    mag_err_new = np.append(mag_err_new, mag_err[:size])
    
    for i, obs in enumerate(mag):
        
        if i < size or i > mag.shape[0] - size:
            continue

        
        med = np.nanmedian(mag[i-size:i+size])
        
        std = np.std(mag[i-size:i+size])
        
        #print('median:', str(med))
        
        if ((obs < med + n*std) and (obs > med - n*std)):
            time_new = np.append(time_new, time[i])
            mag_new = np.append(mag_new, mag[i])
            mag_err_new = np.append(mag_err_new, mag_err[i])
        
        else:
            clipped_count +=1 

    time_new = np.append(time_new, time[-1*size:])
    mag_new = np.append(mag_new, mag[-1*size:])
    mag_err_new = np.append(mag_err_new, mag_err[-1*size:])
    
    if verbose==True:
        print('points removed:', str(clipped_count))

    tb_new = Table()
    tb_new['mag'] = mag_new
    tb_new['mag_err'] = mag_err_new

    ts_new = TimeSeries(tb_new, time=Time(time_new, format='decimalyear'))
    return ts_new



def GP(ts_in, kernel_num, lengthscale):
    """perform gaussian process regression on input timeseries
    Parameters
    ----------

    ts_in : astropy.Timeseries Object
        timeseries to be used for gaussian process regression
    Returns
    -------

    ts: astropy.Timeseries Object
        timeseries containing gaussian process regression results

    hyper_vector: np.ndarray
        array containing information about hyperparameters
    """
   
    x_in = ts_in['time'].to_value('decimalyear')
    y_in = ts_in['mag']
    y_err = ts_in['mag_err']

    # Define range of input space to predict over
    x_min = x_in.min() 
    x_max = x_in.max() 
    
    # Mesh the input space for evaluations of the real function, the prediction and
    # its MSE
    x_space = np.atleast_2d(np.linspace(x_min, x_max, 150)).T
    x_fit = np.atleast_2d(x_in).T
    
    l = (lengthscale[1]-lengthscale[0])/2
    k_RBF = kernels.RBF(length_scale=l, length_scale_bounds=(lengthscale[0], lengthscale[1]))
    k_exp = (kernels.Matern(length_scale=l, length_scale_bounds=(lengthscale[0], lengthscale[1]), nu=0.5))
    k_sine = kernels.ExpSineSquared(length_scale=l, length_scale_bounds=(lengthscale[0], lengthscale[1]), periodicity=1e1, periodicity_bounds=(1e-2, 1e4))
    k_noise = kernels.WhiteKernel(noise_level=l, noise_level_bounds=(lengthscale[0], lengthscale[1]))
    k_matern = (kernels.Matern(length_scale=l, length_scale_bounds=(lengthscale[0], lengthscale[1]), nu=1.5))
    # Matern kernel with nu = 0.5 is equivalent to the exponential kernel
    # Define kernel function
    if kernel_num == 0:
        kernel = 1.0 * k_exp #+ k_noise #+ k_RBF + 1.0*(k_exp*k_sine)
    if kernel_num == 1:
        kernel = 1.0 * k_matern #+ k_noise #NOT GOOD, BLOWS UP A LOT
    if kernel_num == 2:
        kernel = 1.0 * k_sine #+ k_noise
    if kernel_num == 3:
        kernel = 1 * k_RBF
    if kernel_num == 4:
        kernel = (1.0 * k_matern + 1.0 * k_exp ) * 1.0 * k_RBF 
    if kernel_num == 5:
        kernel =  1.0 * k_matern + 1.0 * k_exp #+ k_noise
        
    
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=y_err**2, n_restarts_optimizer=10, normalize_y=True, random_state=1)
    
    # Fit to data using Maximum Likelihood Estimation of the parameters
    gpr.fit(x_fit, y_in)

    # Make the prediction on the meshed x-axis (ask for MSE as well)
    y_pred, y_pred_sigma = gpr.predict(x_space, return_std=True)
    
    # Get log likelihood and hyperparameters
    log_likelihood = gpr.log_marginal_likelihood()
    hyper_params = gpr.kernel_
    
    # ts_avg = aggregate_downsample(ts_in, time_bin_size = 30 * u.day, n_bins=400, aggregate_func=np.nanmean)
    # ts_avg = ts_avg[~ts_avg['mag'].mask]
    # ts_avg['time'] = ts_avg['time_bin_start']

    # cv = LeaveOneOut()
    # scores = cross_val_score(gpr, ts_avg['time'].to_value('decimalyear').reshape(-1, 1), ts_avg['mag'], scoring='neg_mean_absolute_error',cv=cv, n_jobs=-1)
    # RMSE = (np.sqrt(np.mean(np.absolute(scores))))
    # hyper_vector = []
    # hyper_vector.append(log_likelihood)
    # params = hyper_params.get_params()
    # for i, key in enumerate(sorted(params)):
    #     if i in (3,6,10,14,18,20,23):
    #         #print(i, "%s : %s" % (key, params[key]))
    #         hyper_vector.append(params[key])
    
    #compile data into Timeseries
    RMSE=0
    x_space = x_space.flatten()
    y_pred = y_pred.flatten()
    y_pred_sigma = y_pred_sigma.flatten()

    tb = Table()

    tb['mag'] = y_pred
    tb['mag_err'] = y_pred_sigma

    ts = TimeSeries(tb, time=Time(x_space, format='decimalyear'))

    return ts, log_likelihood, hyper_params, RMSE


def match_lightcurves(ts1, ts2):
    """find constant offset between two overlapping regions of two light curves that minimizes the difference between the two
    Parameters
    ----------

    ts1 : astropy.Timeseries Object
        light curve that begins and ends first

    ts2 : astropy.Timeseries Object
        light curve that begins and ends later   
    Returns
    -------

    const: np.float64
        constant offset between the two light curves
    """

    t1 = ts1['time'].to_value('decimalyear')
    m1 = ts1['mag']

    t2 = ts2['time'].to_value('decimalyear')
    m2 = ts2['mag']

    t_diff_start = t1 - t2[0]
    t_diff_end = t2 - t1[-1]

    zero_crossing_start = np.where(np.diff(np.signbit(t_diff_start)))[0]+1
    zero_crossing_end = np.where(np.diff(np.signbit(t_diff_end)))[0]

    if zero_crossing_start.shape[0] == 0 or zero_crossing_end.shape[0] == 0:
        print('using padded times')
        time_diff = (t2[0] - t1[-1])+0.1


        padded_t1 = np.pad(t1, pad_width=(0, 10), mode='linear_ramp', end_values=t1[-1] + time_diff)
        # print(t1[-1] + time_diff)
        padded_t2 = np.pad(t2, pad_width=(10, 0), mode='linear_ramp', end_values=t2[0] - time_diff)
        # print(t2[0] - time_diff)

        t_diff_start = padded_t1 - padded_t2[0]
        t_diff_end = padded_t2 - padded_t1[-1]

        zero_crossing_start = np.where(np.diff(np.signbit(t_diff_start)))[0]+1
        zero_crossing_end = np.where(np.diff(np.signbit(t_diff_end)))[0]

    
    overlap1 = m1[zero_crossing_start[0]:]     
    if zero_crossing_end[0] != 0:              ##edge case where the zero crossing happens at the first element of the second timeseries
        overlap2 = m2[:zero_crossing_end[0]]   
    else:
        overlap2 = m2[:zero_crossing_end[0]+1] ##add 1 because python does not include the nth element in array[:n], here n=0.

    if overlap1.shape[0] < overlap2.shape[0]:
        zero_crossing_end -= (overlap2.shape[0] - overlap1.shape[0])

    if overlap1.shape[0] > overlap2.shape[0]:
        zero_crossing_start += (overlap1.shape[0] - overlap2.shape[0])

    overlap1 = m1[zero_crossing_start[0]:]

    if zero_crossing_end[0] != 0:              ##edge case where the zero crossing happens at the first element of the second timeseries
        overlap2 = m2[:zero_crossing_end[0]]
    else:
        overlap2 = m2[:zero_crossing_end[0]+1] ##add 1 because python does not include the nth element in array[:n], here n=0.

    def minimize_func(c):
        v1, v2 = overlap1, overlap2
        shifted_v2 = v2+c

        return np.sqrt(np.sum(np.square(v1-shifted_v2)))
    
    const = minimize(minimize_func, x0=0).x

    return const
