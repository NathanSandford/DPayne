import sys
sys.path.append('/global/scratch/nathan_sandford/DPayne/')
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import theano.tensor as tt
import pymc3 as pm
from pymc3.backends import HDF5
from pymc3.backends.tracetab import trace_to_dataframe
from scipy.stats import truncnorm

import matplotlib.pyplot as plt
from corner import corner

from DPayne.neural_networks import load_trained_model
from DPayne.utils import scale_labels, rescale_labels

description = \
    '''
    Code trains "The Payne" on synthetic spectra.
    '''

'''
Defaults
'''
base_dir = Path('/global/scratch/nathan_sandford/DPayne')
nn_dir = base_dir.joinpath('neural_nets')
hmc_dir = base_dir.joinpath('hmc_output')


# HMC Settings
elements_to_fit = ['Fe', 'Ca', 'Ni', 'Si', 'Ti', 'Co', 'Mg']
other_to_fit = ['Teff', 'logg', 'v_micro']
priors = {}
random_state = 3457
cores = 24
chains = 24
ntune = 2000
nsamples = 500

'''
Parse Args
'''
parser = argparse.ArgumentParser(description=description)
parser.add_argument("model_name", help="Model name")
parser.add_argument("snr", type=int, help="S/N of the Spectrum")

parser.add_argument("--base_dir", "-bdir", help=f"Base Directory (default: {base_dir.name})")
parser.add_argument("--nn_dir", "-ndir", help=f"Directory of NN files (default: {nn_dir.name})")
parser.add_argument("--hmc_dir", "-hdir", help=f"Output directory for HMC (default: {hmc_dir.name})")


parser.add_argument("--elements_to_fit", "-X", nargs='+',
                    help=f"Elements to fit (default: {' '.join(elements_to_fit)})")
parser.add_argument("--other_to_fit", "-oX", nargs='+',
                    help=f"Other labels to fit (default: {' '.join(other_to_fit)})")
parser.add_argument("--priors", "-p", type=str,
                    help='Dictionary of {"label": sigma} (default: {})')

parser.add_argument("--random_state", "-rand", type=int,
                    help=f"Random state to initialize HMC (default: {random_state})")
parser.add_argument("--ntune", "-Nt", type=int,
                    help=f"Number of tuning steps / chain (default: {ntune})")
parser.add_argument("--nsample", "-Ns", type=int,
                    help=f"Number of samples / chain (default: {nsamples})")
parser.add_argument("--cores", "-co", type=int,
                    help=f"Number of Cores (default: {cores})")
parser.add_argument("--chains", "-ch", type=int,
                    help=f"Number of Chains (default: {chains})")
parser.add_argument("--continue_from_previous", '-cont', action="store_true", default=False,
                    help=f"Continue previous HMC run (default: False)")
args = parser.parse_args()

if args.base_dir:
    base_dir = Path(args.base_dir)
assert base_dir.is_dir(), f'Directory {base_dir} does not exist'

if args.nn_dir:
    nn_dir = base_dir.joinpath(args.nn_dir)
else:
    nn_dir = base_dir.joinpath(nn_dir.name)
assert nn_dir.is_dir(), f'Directory {nn_dir} does not exist'

if args.hmc_dir:
    hmc_dir = base_dir.joinpath(args.hmc_dir)
else:
    hmc_dir = base_dir.joinpath(hmc_dir.name)
assert hmc_dir.is_dir(), f'Directory {hmc_dir} does not exist'

if args.elements_to_fit:
    elements_to_fit = args.elements_to_fit
if args.other_to_fit:
    other_to_fit = args.other_to_fit
if args.priors:
    priors = json.loads(args.priors.replace("\'", '\"'))

if args.random_state:
    random_state = args.random_state
np.random.seed(random_state)
if args.ntune:
    ntune = args.ntune
if args.nsample:
    nsample = args.nsample
if args.cores:
    cores = args.cores
if args.chains:
    chains = args.chains

snr = args.snr
model_name = args.model_name
model_file = nn_dir.joinpath(f'{model_name}_model.pt')
scaling_file = nn_dir.joinpath(f'{model_name}_scaling.npz')
loss_file = nn_dir.joinpath(f'{model_name}_loss.npz')
assert model_file.exists(), f'{model_file} does not exist'
assert scaling_file.exists(), f'{scaling_file} does not exist'
if not loss_file.exists():
    print(f'{loss_file} does not exist. Continuing anyway...')
hmc_trace = hmc_dir.joinpath(f'{model_name}_snr{snr:03d}_trace.h5')
hmc_samples = hmc_dir.joinpath(f'{model_name}_snr{snr:03d}_samples.h5')
hmc_corner = hmc_dir.joinpath(f'{model_name}_snr{snr:03d}_corner.png')
hmc_truespec = hmc_dir.joinpath(f'{model_name}_snr{snr:03d}_spec.npy')

continue_from_previous = args.continue_from_previous
if continue_from_previous:
    assert hmc_trace.exists(), f'{hmc_trace} does not exist'
    print('Not implemented yet, starting from scratch')
else:
    if hmc_trace.exists():
        print(f"Overwriting {hmc_trace}, hope that's okay")
    if hmc_samples.exists():
        print(f"Overwriting {hmc_samples}, hope that's okay")


'''
Load the trained NN
'''
NN_model = load_trained_model(model_name, nn_dir, theano_wrap=True)


'''
Check Model Labels
'''
labels_to_fit = other_to_fit + elements_to_fit
assert set(labels_to_fit) <= set(NN_model.labels),\
    f'{set(labels_to_fit)- set(NN_model.labels)} not label(s) in the model'
assert set(priors.keys()) <= set(NN_model.labels), \
    f'Priors {set(labels_to_fit) - set(NN_model.labels)} not label(s) in the model'
assert set(priors.keys()) <= set(labels_to_fit), \
    f'Priors {set(priors.keys()) - set(labels_to_fit)} not being fit'


'''
Generate Mock Spectrum
'''
theta_true = np.zeros(NN_model.dim_in)
spec_true = NN_model.nn_tt(theta_true).eval()
noisy_spec = spec_true + np.random.normal(0, scale=spec_true/snr, size=spec_true.shape[0])
np.save(hmc_truespec, spec_true)


'''
Scale Priors
'''
tmp = [NN_model.labels.index(item) for item in labels_to_fit]
x_min_fit = NN_model.x_min[tmp]
x_max_fit = NN_model.x_max[tmp]
truth_fit = theta_true[tmp]


'''
Scale Priors
'''
scaled_priors = {}
if priors:
    for label in priors:
        lab_ind = np.argwhere(np.array(labels_to_fit) == label)[0][0]
        lab = rescale_labels(theta_true, NN_model.x_min, NN_model.x_max)[lab_ind] + priors[label]
        scaled_priors[label] = scale_labels(lab, NN_model.x_min, NN_model.x_max)[lab_ind]


'''
Run HMC
'''
with pm.Model() as model:
    # Priors
    theta_list = []
    for label in NN_model.labels:
        if label in labels_to_fit:
            if label in priors:
                theta_list.append(pm.Bound(pm.Normal, lower=-0.5, upper=0.5)(label, mu=0.0, sigma=scaled_priors[label]))
            else:
                theta_list.append(pm.Uniform(label, lower=-0.5, upper=0.5))
        else:
            theta_list.append(0.0)
    theta = tt.stack(theta_list)
    # Model
    model_spec = NN_model.nn_tt(theta)
    # Likelihood
    spec = pm.Normal('spec', mu=model_spec, sd=spec_true/snr, observed=spec_true)
    # Sampling
    backend = HDF5(hmc_trace)
    trace = pm.sample(nsamples, tune=ntune, chains=chains, cores=cores, trace=backend)


samples = pd.DataFrame(columns=labels_to_fit)
samples[labels_to_fit] = trace_to_dataframe(trace, varnames=labels_to_fit)
samples = rescale_labels(samples, x_min_fit, x_max_fit)
samples.to_hdf(hmc_samples, f'SNR={snr}')


'''
Plot Corner Plot
'''
ndim = len(labels_to_fit)
fig = corner(samples, labels=labels_to_fit, truths=rescale_labels(truth_fit, x_min_fit, x_max_fit),
             show_titles=True, quantiles=(0.16, 0.50, 0.84),
             range=list(zip(x_min_fit, x_max_fit)), scale_hist=True,
             max_n_ticks=3,
             label_kwargs=dict(size=24),
             hist_kwargs=dict(density=True))
axes = np.array(fig.axes).reshape((ndim, ndim))
for i, label in enumerate(labels_to_fit):  # Overplot priors
    ax = axes[i, i]
    if label in priors:
        a = (x_min_fit[i] - rescale_labels(truth_fit, x_min_fit, x_max_fit)[i]) / priors[label]
        b = (x_max_fit[i] - rescale_labels(truth_fit, x_min_fit, x_max_fit)[i]) / priors[label]
        x = np.linspace(x_min_fit[i], x_max_fit[i], 1000)
        prior_dist = truncnorm.pdf(x, a, b,
                                   loc=rescale_labels(truth_fit, x_min_fit, x_max_fit)[i],
                                   scale=priors[label])
        ax.plot(x, prior_dist)
plt.savefig(hmc_corner)
