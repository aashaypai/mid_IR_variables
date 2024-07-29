import emcee
import corner

import numpy as np
import matplotlib.pyplot as plt

class MCMC:

    def __init__(self, optical_data, IR_data):

        self.t_opt = optical_data['time'].to_value('decimalyear')
        self.m_opt = optical_data['mag']
        self.err_opt = optical_data['mag_err']


        self.t_wise = IR_data['time'].to_value('decimalyear')
        self.m_wise = IR_data['mag']
        self.err_wise = IR_data['mag_err']

    def hat(self, width):

        y = np.ones(width)
        y[0], y[-1] = 0, 0

        return y/np.nansum(y)


    def convolve(self, lag):

        kern_width = int(np.round(0.5*lag/np.nanmean(np.diff(self.t_opt)), 0))
        #print(kern_width)
        kern = self.hat(kern_width)

        conv = np.convolve(kern, self.m_opt, mode='valid')
        t_conv = self.t_opt[(kern_width-1)//2:-(kern_width-1)//2]
        err_conv = self.err_opt[(kern_width-1)//2:-(kern_width-1)//2]
        return conv, t_conv, err_conv
        
    def model(self, model_params):
        #print(model_params[0])
        c, t_c, err_c = self.convolve(model_params[0])
        m = model_params[1] * c + model_params[2]

        inds = np.abs(self.t_wise[:, None] - (t_c[None, :]+model_params[0])).argmin(axis=-1)

        mags = m[inds]
        errs = err_c[inds]
        
        return mags, errs

    def lnlikelihood(self, model_params):
        
        # c, t_c, err_c = convolve(model_params[0])
        # m = model_params[1] * c + model_params[2]

        # inds = np.abs(wise_obj_p1['time'].to_value('decimalyear')[:, None] - (t_c[None, :]+model_params[0])).argmin(axis=-1)
        
        # lnlike = -0.5 * (np.sum(((m[inds]-wise_obj_p1['mag'])/err_c[inds])**2))
        mags, errs = self.model(model_params)
        
        lnlike = -0.5 * (np.sum(((mags-self.m_wise)/errs)**2))

        return lnlike

    def lnprior(self, model_params):

        lag, amp, const = model_params[0], model_params[1], model_params[2]

        if (self.ranges[0][0]<lag<self.ranges[0][1] and self.ranges[1][0]<amp<self.ranges[1][1]  and self.ranges[2][0]<const<self.ranges[2][1]):
            return 0.
        else:
            return -np.inf
        
    def lnprob(self, model_params):
        lp = self.lnprior(model_params)

        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlikelihood(model_params)

    def main(self, p0, nwalkers, niter, ndim):
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob)

        print("Running burn-in...")
        p0, _, _ = sampler.run_mcmc(p0, 100)
        sampler.reset()

        print("Running production...")
        pos, prob, state = sampler.run_mcmc(p0, niter)

        return sampler, pos, prob, state
    
    def run_MCMC(self, kwargs):
        self.initial = kwargs.get("initial")
        self.nwalkers = kwargs.get("nwalkers", 100)
        self.ndim = kwargs.get("ndim", 3)
        self.niter = kwargs.get("niter", 500)
        self.ranges = kwargs.get("ranges", ((0.04, 1), (0.1, 20), (-100, 0)))

        default_p0 = [np.array(self.initial[1:]) + 1e-1 * np.random.randn(2) for i in range(self.nwalkers)]
        default_p0 = np.insert(default_p0, 0, np.random.normal(loc=0.5, scale=0.15,size=len(default_p0)), axis=1)

        self.p0 = kwargs.get("p0", default_p0)

        self.sampler, self.pos, self.prob, self.state = self.main(self.p0, self.nwalkers, self.niter, self.ndim)

    def generate_corner_plot(self, CORNER_KWARGS):
        samples = self.sampler.flatchain

        fig = corner.corner(samples, **CORNER_KWARGS)
        fig.subplots_adjust(right=1.5, top=1.5)

    def generate_walker_path_plot(self, kwargs):
        alpha=kwargs.get("alpha", 0.1)
        color = kwargs.get("color", "k")

        fig, axes = plt.subplots(self.ndim, figsize=(10, 7), sharex=True)
        samples = self.sampler.get_chain()

        labels = kwargs.get("labels", [r'$\Delta t$', 'AMP', 'const'])
        for i in range(self.ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], color=color, alpha=alpha)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number")

    def generate_monte_carlos_plot(self, kwargs):
        alpha=kwargs.get("alpha", 0.1)
        num=kwargs.get("num", 100)
        color = kwargs.get("color", "r")

        fig, ax = plt.subplots()
        samples = self.sampler.flatchain
        for theta in samples[np.random.randint(len(samples), size=num)]:

            #print(theta)
            if theta[0]<0:
                continue
            conv, t_conv, conv_err = self.convolve(theta[0])

            ax.plot(t_conv+theta[0], theta[1] * conv + theta[2], color=color, alpha=alpha)
    
        ax.set_ylabel('mag')
        ax.set_xlabel('date')
        ax.invert_yaxis()
        

    def set_optical_data(self, optical_data):
        self.t_opt = optical_data['time'].to_value('decimalyear')
        self.m_opt = optical_data['mag']
        self.err_opt = optical_data['mag_err']
    
    def set_IR_data(self, IR_data):
        self.t_wise = IR_data['time'].to_value('decimalyear')
        self.m_wise = IR_data['mag']
        self.err_wise = IR_data['mag_err']

    