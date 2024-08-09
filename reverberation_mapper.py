#three ways of finding time lag.from tqdm import tqdm

import numpy as np

import matplotlib
import matplotlib.pyplot as plt


from mid_IR_variables import fileIO_processing as fp
from mid_IR_variables import lightcurve_processing as lp

from scipy import optimize
from scipy.stats import pearsonr

SAVE_FILEPATH = 'C:/Users/paiaa/Documents/Research/Blanton Lab/Midir Variables/Figures/'
IMPORT_FILEPATH ='C:/Users/paiaa/Documents/Research/Blanton Lab/Midir Variables/'

mnsa_hdu, manga_wise_hdu, pipe3d_hdu = fp.import_manga(6, 1, 1)
mnsa, mwv, pipe3d = mnsa_hdu.data, manga_wise_hdu.data, pipe3d_hdu.data

class FixedWidth_Model:

    def __init__(self, plateifu, optical_data=None, w1_data=None, w2_data=None):

        self.verbose = False
        self.plateifu = plateifu
        self.wise_band = None
        #self.variable_kern_width = False

        if optical_data == None:

            try:
                self.optical_data, self.optical_GP = self.generate_optical_lightcurve()
                
                self.t_opt = self.optical_GP['time'].to_value('decimalyear')
                self.m_opt = self.optical_GP['mag']
                self.err_opt = self.optical_GP['mag_err']
                print('**Optical Data and GP Generated**')
            except:
                print('**Error: Unable to Generate Optical Data**')

        else:
            self.optical_GP = optical_data
            
            self.t_opt = self.optical_GP['time'].to_value('decimalyear')
            self.m_opt = self.optical_GP['mag']
            self.err_opt = self.optical_GP['mag_err']

        if w1_data == None or w2_data == None:

            try:
                self.w1, self.w2 = self.generate_wise_lightcurve()

                self.t_w1 = self.w1['time'].to_value('decimalyear')
                self.m_w1 = self.w1['mag']
                self.err_w1 = self.w1['mag_err']

                self.t_w2 = self.w2['time'].to_value('decimalyear')
                self.m_w2 = self.w2['mag']
                self.err_w2 = self.w2['mag_err']
                print('**IR Data Generated**')

            except:
                print('**Error: Unable to Generate IR Data**')

        else:
            self.w1, self.w2 = w1_data, w2_data

            self.t_w1 = self.w1['time'].to_value('decimalyear')
            self.m_w1 = self.w1['mag']
            self.err_w1 = self.w1['mag_err']

            self.t_w2 = self.w2['time'].to_value('decimalyear')
            self.m_w2 = self.w2['mag']
            self.err_w2 = self.w2['mag_err']

        self.preds_w1, self.errs_w1, self.chisq_w1, self.weightedchisq_w1, self.model1, self.const1, self.amp1, self.width1, self.lag1, self.P1 = [], [], [], [], [], [], [], [], [], []
        self.preds_w2, self.errs_w2, self.chisq_w2, self.weightedchisq_w2, self.model2, self.const2, self.amp2, self.width2, self.lag2, self.P2 = [], [], [], [], [], [], [], [], [], []

    def generate_wise_lightcurve(self):

        mnsa_hdu, manga_wise_hdu, pipe3d_hdu = fp.import_manga(6, 1, 1)
        mnsa, mwv, pipe3d = mnsa_hdu.data, manga_wise_hdu.data, pipe3d_hdu.data 

        w1 = fp.process_wise(self.plateifu, mwv, band=1)
        w2 = fp.process_wise(self.plateifu, mwv, band=2)

        return w1, w2

    def generate_optical_lightcurve(self, l=[0.95, 1.05]):

        optical_lightcurve = lp.generate_combined_lightcurve(pifu=self.plateifu)

        poly_subtracted_obj_p, fit, fitted_poly = lp.polyfit_lightcurves(optical_lightcurve, deg=10)
        gp, llh, hyperparams, cov = lp.GP(poly_subtracted_obj_p, kernel_num=3, lengthscale=(l[0], l[1]))
        gp_fitted_poly =lp.make_polynomial(gp, fit)
        
        gp['mag']+=gp_fitted_poly
        print('**GP Kernel:', str(hyperparams)+'**')

        return optical_lightcurve, gp


    def hat(self, width):

        y = np.ones(width)

        y[0], y[-1] = 0, 0

        return y/np.nansum(y)


    def convolve(self, lag, width):
        if (-0.01<lag<0.01): #-0.02<=lag<=0.02
            #print('no convolution')
            conv, t_conv, err_conv = (self.m_opt-np.nanmedian(self.m_opt)), self.t_opt, self.err_opt
            return conv, t_conv, err_conv
        #if width*lag < 0.55:

        kern_width = int(np.round(width*np.abs(lag)/np.nanmean(np.diff(self.t_opt)), 0))
            
        #print(kern_width)
        kern = self.hat(kern_width)

        conv = np.convolve(kern, self.m_opt-np.nanmedian(self.m_opt), mode='valid')
        t_conv = self.t_opt[(kern_width-1)//2:-(kern_width-1)//2]
        err_conv = self.err_opt[(kern_width-1)//2:-(kern_width-1)//2]

        return conv, t_conv, err_conv
    
    def fit_linear(self, conv, IR_data):

        def offset(x):
            a, c = x[0], x[1]
            conv_new = a * conv + c

            difference = conv_new - IR_data
            return difference
        
        amp, const = optimize.leastsq(offset, x0=[1, -50])

        return amp, const
    
    
    def weight_function(self, conv, inds):

        num_last_element = np.size(inds[inds==np.size(conv)-1])
        P = (np.count_nonzero(inds)-num_last_element)/np.size(inds)
        #print(np.size(inds), np.count_nonzero(inds), num_last_element)
        return np.square(P)
    
    def predict_mags(self, params):
        
        lag = params[0]#width=, params[1]
        width=0.5
        #width = self.check_width(lag, width)

        if self.wise_band == 1:
            IR_data = self.w1
        if self.wise_band == 2:
            IR_data = self.w2

        if (-0.01<lag<0.01): #-0.02<=lag<=0.02
            #print('no convolution')
            conv, t_conv, err_conv = (self.m_opt-np.nanmedian(self.m_opt)), self.t_opt, self.err_opt
            inds = np.abs(IR_data['time'].to_value('decimalyear')[:, None] - (t_conv[None, :])).argmin(axis=-1)
        else:
            conv, t_conv, err_conv = self.convolve(lag, width)
            inds = np.abs(IR_data['time'].to_value('decimalyear')[:, None] - (t_conv[None, :]+lag)).argmin(axis=-1)

        
        amp, const = self.fit_linear(conv[inds], IR_data['mag'])[0]

        model = amp * conv + const

        if self.verbose == True: #verbose option
            print(lag, amp, const)

        predicted_mags = model[inds]
        predicted_errs = err_conv[inds]

        P = self.weight_function(conv, inds)
        if self.wise_band == 1:
            self.preds_w1.append(predicted_mags)
            self.errs_w1.append(predicted_errs)
            self.model1.append(model)
            self.const1.append(const)
            self.amp1.append(amp)
            self.width1.append(width)
            self.lag1.append(lag)
            self.P1.append(P)
        if self.wise_band == 2:
            self.preds_w2.append(predicted_mags)
            self.errs_w2.append(predicted_errs)
            self.model2.append(model)
            self.const2.append(const)
            self.amp2.append(amp)
            self.width2.append(width)
            self.lag2.append(lag)
            self.P2.append(P)
        return predicted_mags, predicted_errs, amp, const, P

    def chisq(self, params):
        if self.wise_band == 1:
            IR_data = self.w1
        if self.wise_band == 2:
            IR_data = self.w2
            
        model_mags, model_errs, _, _, P = self.predict_mags(params)

        #norm_model_mags, norm_IR_data = model_mags - np.nanmean(model_mags), IR_data['mag'] - np.nanmean(IR_data['mag'])
        chisq = np.sum(((model_mags-IR_data['mag'])/model_errs)**2) #IR_data['mag_err']
        
        if self.wise_band == 1:
            self.chisq_w1.append(chisq)
            self.weightedchisq_w1.append(chisq/P)
        if self.wise_band == 2:
            self.chisq_w2.append(chisq)
            self.weightedchisq_w2.append(chisq/P)
        return chisq/P


    def minimize_chisq(self, kwargs):
        
        m = kwargs.get("model")
        x0 = kwargs.get("x0", [0.2, 1., 1.])
        niter = kwargs.get("niter", 100)
        T = kwargs.get("T", 1)
        stepsize = kwargs.get("stepsize", 0.5)
        ranges = kwargs.get("ranges")
        
        self.wise_band = kwargs.get("wise band")
        self.verbose = kwargs.get("verbose", False)

        if self.verbose == True:
            print(kwargs)
            
        if self.wise_band == 1:
            self.preds_w1, self.errs_w1, self.chisq_w1, self.weightedchisq_w1, self.model1, self.const1, self.amp1, self.width1, self.lag1, self.P1 = [], [], [], [], [], [], [], [], [], []
        if self.wise_band == 2:
            self.preds_w2, self.errs_w2, self.chisq_w2, self.weightedchisq_w2, self.model2, self.const2, self.amp2, self.width2, self.lag2, self.P2 = [], [], [], [], [], [], [], [], [], []

        if m == 'brute':
            model = optimize.brute(self.chisq, ranges=ranges, full_output=True, disp=True, finish=None)  ## finish=optimize.minimize

        if m == 'basinhopping':
            minimizer_kwargs = { "method": "L-BFGS-B","bounds": ranges}
            model = optimize.basinhopping(self.chisq, x0=x0, stepsize=stepsize, niter=niter, T=T,minimizer_kwargs=minimizer_kwargs, callback=print_fun)

        if m == 'dualannealing':
            def print_fun(x, f, accepted):
                print("at minimum", str(x),  "with chisq:", str(f), "accepted:", int(accepted))

            #minimizer_kwargs = { "method": "L-BFGS-B","bounds": ranges}
            model = optimize.dual_annealing(self.chisq, bounds=ranges, callback=print_fun)

        #model = optimize.minimize(chisq, method='Nelder-Mead', x0=[0.2, 1, 1], bounds=((0.1, 5), (-np.inf,np.inf), (-np.inf, np.inf)))
        #ranges=(slice(0.01, 1, 0.01), slice(-100, 100, 0.1), slice(-100, 100, 0.1))
    
        return model
    ########################GETTERS and SETTERS########################

    def get_plateifu(self):
        return self.plateifu
    
    def set_plateifu(self, pifu):
        self.plateifu = pifu

    def get_wise_band(self):
        return self.wise_band
    
    def set_wise_band(self, wband):
        self.wise_band = wband

    def get_verbose(self):
        return self.verbose
    
    def set_verbose(self, v):
        self.verbose = v
        
    def get_bestfitparams(self, wise_band):
        if wise_band == 1:
            i = np.where(self.chisq_w1 == np.nanmin(self.chisq_w1))[0][0]
            vals = np.array([self.lag1[i], self.width1[i], self.amp1[i], self.const1[i]])

        if wise_band == 2:
            i = np.where(self.chisq_w2 == np.nanmin(self.chisq_w2))[0][0]
            vals = np.array([self.lag2[i], self.width2[i], self.amp2[i], self.const2[i]])

        return vals

    def get_bfp_index(self, wise_band):
        if wise_band == 1:
            i = np.where(self.chisq_w1 == np.nanmin(self.chisq_w1))[0][0]
        if wise_band == 2:
            i = np.where(self.chisq_w2 == np.nanmin(self.chisq_w2))[0][0]

        return i
    
class VariableWidth_Model(FixedWidth_Model):
    
    def check_width(self, lag, width, cadence=0.55):

        if lag * width < cadence:
            return width
        
        else:
            return (cadence-0.01)/lag
        
    def predict_mags(self, params):
        
        lag, width = params[0], params[1]
        width = self.check_width(lag, width)

        if self.wise_band == 1:
            IR_data = self.w1
        if self.wise_band == 2:
            IR_data = self.w2

        if (-0.01<lag<0.01): #-0.02<=lag<=0.02
            #print('no convolution')
            conv, t_conv, err_conv = (self.m_opt-np.nanmedian(self.m_opt)), self.t_opt, self.err_opt
            inds = np.abs(IR_data['time'].to_value('decimalyear')[:, None] - (t_conv[None, :])).argmin(axis=-1)
        else:
            conv, t_conv, err_conv = self.convolve(lag, width)
            inds = np.abs(IR_data['time'].to_value('decimalyear')[:, None] - (t_conv[None, :]+lag)).argmin(axis=-1)

        
        amp, const = self.fit_linear(conv[inds], IR_data['mag'])[0]

        model = amp * conv + const

        if self.verbose == True: #verbose option
            print(lag, amp, const)

        predicted_mags = model[inds]
        predicted_errs = err_conv[inds]
        print(predicted_errs)
        if self.wise_band == 1:
            self.preds_w1.append(predicted_mags)
            self.errs_w1.append(predicted_errs)
            self.model1.append(model)
            self.const1.append(const)
            self.amp1.append(amp)
            self.width1.append(width)
            self.lag1.append(lag)
        if self.wise_band == 2:
            self.preds_w2.append(predicted_mags)
            self.errs_w2.append(predicted_errs)
            self.model2.append(model)
            self.const2.append(const)
            self.amp2.append(amp)
            self.width2.append(width)
            self.lag2.append(lag)
        return predicted_mags, predicted_errs, amp, const

    def chisq(self, params):
        if self.wise_band == 1:
            IR_data = self.w1
        if self.wise_band == 2:
            IR_data = self.w2
            
        model_mags, model_errs, _, _ = self.predict_mags(params)

        #norm_model_mags, norm_IR_data = model_mags - np.nanmean(model_mags), IR_data['mag'] - np.nanmean(IR_data['mag'])
        chisq = np.sum(((model_mags-IR_data['mag'])/model_errs)**2) #IR_data['mag_err']
        
        if self.wise_band == 1:
            self.chisq_w1.append(chisq)
        if self.wise_band == 2:
            self.chisq_w2.append(chisq)

        return chisq


    def minimize_chisq(self, kwargs):
        
        m = kwargs.get("model")
        x0 = kwargs.get("x0", [0.2, 1., 1.])
        niter = kwargs.get("niter", 100)
        T = kwargs.get("T", 1)
        stepsize = kwargs.get("stepsize", 0.5)
        ranges = kwargs.get("ranges")
        
        self.wise_band = kwargs.get("wise band")
        self.verbose = kwargs.get("verbose", False)

        if self.verbose == True:
            print(kwargs)
            
        if self.wise_band == 1:
            self.preds_w1, self.errs_w1, self.chisq_w1, self.model1, self.const1, self.amp1, self.width1, self.lag1 = [], [], [], [], [], [], [], []
        if self.wise_band == 2:
            self.preds_w2, self.errs_w2, self.chisq_w2, self.model2, self.const2, self.amp2, self.width2, self.lag2 = [], [], [], [], [], [], [], []

        if m == 'brute':
            model = optimize.brute(self.chisq, ranges=ranges, full_output=True, disp=True, finish=None)  ## finish=optimize.minimize

        if m == 'basinhopping':
            minimizer_kwargs = { "method": "L-BFGS-B","bounds": ranges}
            model = optimize.basinhopping(self.chisq, x0=x0, stepsize=stepsize, niter=niter, T=T,minimizer_kwargs=minimizer_kwargs, callback=print_fun)

        if m == 'dualannealing':
            def print_fun(x, f, accepted):
                print("at minimum", str(x),  "with chisq:", str(f), "accepted:", int(accepted))

            #minimizer_kwargs = { "method": "L-BFGS-B","bounds": ranges}
            model = optimize.dual_annealing(self.chisq, bounds=ranges, callback=print_fun)

        #model = optimize.minimize(chisq, method='Nelder-Mead', x0=[0.2, 1, 1], bounds=((0.1, 5), (-np.inf,np.inf), (-np.inf, np.inf)))
        #ranges=(slice(0.01, 1, 0.01), slice(-100, 100, 0.1), slice(-100, 100, 0.1))
    
        return model
    # def minimize_chisq(self, kwargs):
        
    #     m = kwargs.get("model")
    #     x0 = kwargs.get("x0", [0.2, 1., 1.])
    #     niter = kwargs.get("niter", 100)
    #     T = kwargs.get("T", 1)
    #     stepsize = kwargs.get("stepsize", 0.5)
    #     self.ranges = kwargs.get("ranges")
        
    #     self.wise_band = kwargs.get("wise band")
    #     self.verbose = kwargs.get("verbose", False)

    #     if self.verbose == True:
    #         print(kwargs)
            
    #     if self.wise_band == 1:
    #         self.preds_w1, self.chisq_w1, self.model1, self.const1, self.amp1, self.width1, self.lag1 = [], [], [], [], [], [], []
    #     if self.wise_band == 2:
    #         self.preds_w2, self.chisq_w2, self.model2, self.const2, self.amp2, self.width2, self.lag2 = [], [], [], [], [], [], []

    #     if m == 'brute':
    #         model = optimize.brute(self.chisq, ranges=self.ranges, full_output=True, disp=True, finish=None)  ## finish=optimize.minimize

    #     if m == 'basinhopping':
    #         minimizer_kwargs = { "method": "L-BFGS-B","bounds": self.ranges}
    #         model = optimize.basinhopping(self.chisq, x0=x0, stepsize=stepsize, niter=niter, T=T,minimizer_kwargs=minimizer_kwargs, callback=print_fun)

    #     if m == 'dualannealing':
    #         def print_fun(x, f, accepted):
    #             print("at minimum", str(x),  "with chisq:", str(f), "accepted:", int(accepted))

    #         #minimizer_kwargs = { "method": "L-BFGS-B","bounds": ranges}
    #         model = optimize.dual_annealing(self.chisq, bounds=self.ranges, callback=print_fun)
    #     #model = optimize.minimize(chisq, method='Nelder-Mead', x0=[0.2, 1, 1], bounds=((0.1, 5), (-np.inf,np.inf), (-np.inf, np.inf)))
    #     #ranges=(slice(0.01, 1, 0.01), slice(-100, 100, 0.1), slice(-100, 100, 0.1))
        
    #     filtered_model = self.filter_results(model=model)
    #     return filtered_model, model

    # def filter_results(self, model, cadence=0.55):
    #     chisq_vals = model[3].T.copy()
    #     #print('yes')
    #     lag_ranges = np.arange(self.ranges[0][0], self.ranges[0][1], self.ranges[0][2])
    #     width_ranges = np.arange(self.ranges[1][0], self.ranges[1][1], self.ranges[1][2])
    #     #print(lag_ranges, width_ranges)
        
    #     grid = np.outer(width_ranges, np.abs(lag_ranges)) #find width*lag values, need to be smaller than cadence of 0.55 yrs

    #     mask = grid<cadence

    #     chisq_vals[~mask] = 1e5

    #     width_min = np.where(chisq_vals == np.nanmin(chisq_vals))[0][0]
    #     lag_min = np.where(chisq_vals == np.nanmin(chisq_vals))[1][0]
        
    #     #flattened_ind = lag_min*width.shape[0]+width_min
    #     lag_val, width_val, chisq_val = lag_ranges[lag_min], width_ranges[width_min], chisq_vals[width_min, lag_min]
    #     pred_mags, pred_errs, amp_val, const_val = self.predict_mags(params=[lag_val, width_val])
        
    #     filtered_mins = np.array([lag_val, width_val, amp_val, const_val])

    #     if self.verbose == True:
    #         print('minima:', filtered_mins, chisq_val, '(chi^2)')
        
    #     return filtered_mins, chisq_val


####################################################################################################################################################
def ccf(optical_data, IR_data, wise_band):
    
    if wise_band == 1:
        m_accretion_disk = optical_data['mag'] * (0.16)**(1/3) ##Lyu 2019 paper, (nu_IR/nu_OPT)^(1/3) for W1

    if wise_band == 2:
        m_accretion_disk = optical_data['mag'] * (0.12)**(1/3) ##Lyu 2019 paper, (nu_IR/nu_OPT)^(1/3) for W2

    inds = np.abs(IR_data['time'].to_value('decimalyear')[:, None] - (optical_data['time'].to_value('decimalyear')[None, :])).argmin(axis=-1)
    ind_min = inds[0]
    #print(inds)
    i = 0

    delta_t = np.mean(np.diff(optical_data['time'].to_value('decimalyear')))
    #print(delta_t)
    ccf = np.array([])
    time_lag = np.array([])

    IR_mags_norm, opt_mags_norm = IR_data['mag']-np.nanmean(IR_data['mag']), optical_data['mag']-np.nanmean(optical_data['mag'])
    while ind_min > 0:
        pearson_coeff= pearsonr(IR_mags_norm, opt_mags_norm[inds-i])[0]
        ccf = np.append(ccf, pearson_coeff)
        #print(pearson_coeff)
        lag = i * delta_t
        time_lag = np.append(time_lag, lag)
        
        ind_min = (inds-i)[0]
        i+=1
    return ccf, time_lag





