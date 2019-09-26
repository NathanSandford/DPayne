import numpy as np
from scipy.optimize import curve_fit
from .utils import doppler_shift


def fit_normalized_spec(spectrum, NN, p0=None, num_p0=1, tol=5e-4, RV_bounds=(-30, 30)):
    """

    :param spectrum:
    :param NN:
    :param p0:
    :param num_p0:
    :param tol:
    :param RV_bounds:
    :return:
    """
    wavelength = spectrum.wavelength

    # Define Model Function
    def fit_func(dummy_variable, *labels):
        spec = NN.spectrum(labels[:-1], scaled=True, tensor_safe=False)
        return doppler_shift(wavelength, spec, labels[-1])

    # Set Bounds
    bounds = (np.append(NN.scale_labels(NN.NN_Coeffs[-2]), [RV_bounds[0]]),
              np.append(NN.scale_labels(NN.NN_Coeffs[-1]), [RV_bounds[1]]))

    # Initialize Walkers
    if p0 is None:
        p0 = np.zeros(len(NN.labels))
    # all_p0 = np.random.normal(p0, scale=0.5, size=(num_p0, p0.shape[0]))
    all_p0 = np.random.uniform(low=bounds[0], high=bounds[1], size=(num_p0, p0.shape[0]))


    # Run Optimizer
    popt, pcov, model_spec = fit_all_p0s(fit_func=fit_func,
                                         norm_spec=spectrum.flux,
                                         spec_err=spectrum.err,
                                         all_p0=all_p0,
                                         bounds=bounds,
                                         tol=tol)
    return popt, pcov, model_spec


def fit_all_p0s(fit_func, norm_spec, spec_err, all_p0, bounds, tol=5e-4):
    """

    :param fit_func:
    :param norm_spec:
    :param spec_err:
    :param all_p0:
    :param bounds:
    :param tol:
    :return:
    """
    all_popt, all_chi2, all_model_specs, all_pcov = [], [], [], []
    for i, x0 in enumerate(all_p0):
        try:
            popt, pcov = curve_fit(fit_func, xdata=[], ydata=norm_spec,
                                   sigma=spec_err, p0=x0, bounds=bounds,
                                   ftol=tol, xtol=tol,
                                   absolute_sigma=True, method='trf')
            model_spec = fit_func([], *popt)
            chi2 = np.sum((model_spec - norm_spec)**2/spec_err)
        # Failed to converge (should not happen for a simple model)
        except RuntimeError:
            popt, pcov = x0, np.zeros((len(x0), len(x0)))
            model_spec = np.copy(norm_spec)
            chi2 = np.inf
        all_popt.append(popt)
        all_chi2.append(chi2)
        all_model_specs.append(model_spec)
        all_pcov.append(pcov)
    all_popt = np.array(all_popt)
    all_chi2 = np.array(all_chi2)
    all_model_specs = np.array(all_model_specs)
    all_pcov = np.array(all_pcov)
    # Choose best model
    best = np.argmin(all_chi2)
    popt, pcov, model_spec \
        = all_popt[best], all_pcov[best], all_model_specs[best]
    return popt, pcov, model_spec
