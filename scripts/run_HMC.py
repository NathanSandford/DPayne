import sys
sys.path.append('/global/scratch/nathan_sandford/DPayne/')
import numpy as np
import pandas as pd
import pymc3 as pm
from pymc3.backends import HDF5
from pymc3.backends.tracetab import trace_to_dataframe
import theano.tensor as tt
from DPayne.neuralnetworks_deprecated import NeuralNet, sigmoid
from corner import corner
import matplotlib.pyplot as plt

# Directories
kout = '/clusterfs/dweisz/nathan_sandford/kurucz_out/'
nn_dir = kout + 'NN_results/'
dpayne_dir = '/global/scratch/nathan_sandford/DPayne/'
hmc_out = dpayne_dir + 'scripts/HMC_samples/'

# HMC Settings
nn_file = nn_dir + sys.argv[1]
SNR = int(sys.argv[2])
samples_file = hmc_out + nn_file[:-4] + '_samples.h5'
cores = 24
chains = 24
nsamples = 250
ntune = 750

# Load Neural Net
NN = NeuralNet(nn_file, nhidden=2, training_method='all_pix')
w0, w1, w2, b0, b1, b2, x_min, x_max = NN.NN_Coeffs

# Set Labels to Fit
labels_to_fit = ['Teff', 'logg', 'Fe', 'Ca', 'Ni', 'Si', 'Ti', 'Co', 'Mg']

# Generate Mock Spectrum
theta_true = np.zeros(NN.labels.shape)
spec_true = NN.spectrum(theta_true, scaled=True)
spec_true += 1/SNR * spec_true * np.random.normal(size=spec_true.shape)

with pm.Model() as model:
    # Priors
    theta_list = []
    for label in NN.labels:
        if label in labels_to_fit:
            theta_list.append(pm.Uniform(label, lower=-0.5, upper=0.5))
        else:
            theta_list.append(0.0)
    theta = tt.stack(theta_list)
    # Model
    inside = sigmoid(tt.tensordot(a=w0, b=theta, axes=1) + b0)
    middle = sigmoid(tt.tensordot(a=w1, b=inside, axes=1) + b1)
    outside = tt.tensordot(a=w2, b=middle, axes=1) + b2
    model_spec = pm.Deterministic('model_spec', outside)
    # Likelihood
    spec = pm.Normal('spec', mu=model_spec, sd=1/SNR, observed=spec_true)
    # Sampling
    backend = HDF5('../NN_data/trace.h5')
    trace = pm.sample(nsamples, tune=ntune, chains=chains, cores=cores, trace=backend)

samples = pd.DataFrame(columns=NN.labels)
samples[labels_to_fit] = trace_to_dataframe(trace, varnames=labels_to_fit)
samples = NN.rescale_labels(samples)
samples.to_hdf(samples_file, f'SNR={SNR}')

fig = corner(samples[labels_to_fit], labels=labels_to_fit, show_titles=True)
plt.savefig(f'{nn_file[:-4]}_{SNR}.png')
