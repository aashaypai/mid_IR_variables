#three ways of finding time lag.from tqdm import tqdm

import numpy as np

import matplotlib
import matplotlib.pyplot as plt


from mid_IR_variables import fileIO_processing as fp
from mid_IR_variables import lightcurve_processing as lp

from scipy import optimize


SAVE_FILEPATH = 'C:/Users/paiaa/Documents/Research/Blanton Lab/Midir Variables/Figures/'
IMPORT_FILEPATH ='C:/Users/paiaa/Documents/Research/Blanton Lab/Midir Variables/'

mnsa_hdu, manga_wise_hdu, pipe3d_hdu = fp.import_manga(6, 1, 1)
mnsa, mwv, pipe3d = mnsa_hdu.data, manga_wise_hdu.data, pipe3d_hdu.data

class reverberation_mapper:

    def __init__(self, plateifu, optical_data=None, w1_data=None, w2_data=None):

        self.verbose = False
        self.plateifu = plateifu
        self.wise_band = None

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

        self.preds_w1, self.chisq_w1 = [], []
        self.preds_w2, self.chisq_w2 = [], []

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


    def convolve(self, lag):

        kern_width = int(np.round(2*lag/np.nanmean(np.diff(self.t_opt)), 0))
        #print(kern_width)
        kern = self.hat(kern_width)

        conv = np.convolve(kern, self.m_opt, mode='valid')
        t_conv = self.t_opt[(kern_width-1)//2:-(kern_width-1)//2]

        return conv, t_conv
            
    def predict_mags(self, params):
        
        lag, amp, const = params[0], params[1], params[2]
        if self.wise_band == 1:
            IR_data = self.w1
        if self.wise_band == 2:
            IR_data = self.w2

        if self.verbose == True: #verbose option
            print(lag, amp, const)

        conv, t_conv = self.convolve(lag)

        inds = np.abs(IR_data['time'].to_value('decimalyear')[:, None] - (t_conv[None, :]+lag)).argmin(axis=-1)
        #print(inds)
        model = amp * conv + const

        predicted_mags = model[inds]
        predicted_errs = self.err_opt[inds]

        if self.wise_band == 1:
            self.preds_w1.append(predicted_mags)
        if self.wise_band == 2:
            self.preds_w2.append(predicted_mags)

        return predicted_mags, predicted_errs

    def chisq(self, params):
        if self.wise_band == 1:
            IR_data = self.w1
        if self.wise_band == 2:
            IR_data = self.w2
            
        model_mags, model_errs = self.predict_mags(params)

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
            self.preds_w1, self.chisq_w1 = [], []
        if self.wise_band == 2:
            self.preds_w2, self.chisq_w2 = [], []

        if m == 'brute':
            model = optimize.brute(self.chisq, ranges=ranges, full_output=True, finish=None, disp=True)

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


