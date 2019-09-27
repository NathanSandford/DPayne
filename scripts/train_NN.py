import sys
sys.path.append('/global/scratch/nathan_sandford/DPayne/')
from pathlib import Path
import numpy as np
import pandas as pd
from DPayne.utils import split_data
from The_Payne import training

input_dir = Path('/global/scratch/nathan_sandford/kurucz/kurucz_out/synthetic_spectra/HMC_valid/R6500')
nn_default_file = Path('NN_normalized_spectra.pt')
NN_output_file = Path('/global/scratch/nathan_sandford/DPayne/NN_data/NN_R6500.pt')
model_param_output_file = Path('/global/scratch/nathan_sandford/DPayne/NN_data/NN_R6500_par.npz')

# Sample Settings
frac_train = 0.75
randomize = True
random_state = 3457
element_labels = ['Fe', 'Ca', 'Ni', 'Si', 'Ti', 'Co', 'Mg']

labels_to_train_on = ['Teff', 'logg', 'v_micro'] + element_labels

# NN Settings
num_neurons = 300
num_steps = 1e3 #1e5
num_features = 64*5
mask_size = 11
learning_rate = 0.001


spec_files = [spec_file for spec_file in input_dir.iterdir()]
spectra = pd.DataFrame()
labels = pd.DataFrame()
for spec_file in spec_files:
    spectra = pd.concat([spectra, pd.read_hdf(spec_file, 'spectra')], axis=1)
    labels = pd.concat([labels, pd.read_hdf(spec_file, 'labels')], axis=1)
spectra.columns = np.arange(spectra.shape[1])
labels.columns = np.arange(labels.shape[1])
labels.loc[set(labels.index) ^ {'Teff', 'logg', 'v_micro', 'Fe'}] -= labels.loc['Fe']

training_labels, training_spectra, validation_labels, validation_spectra = split_data(spectra, labels,
                                                                                      labels_to_train_on,
                                                                                      frac_train=frac_train,
                                                                                      randomize=randomize,
                                                                                      random_state=random_state)

training_loss, validation_loss = training.neural_net(training_labels, training_spectra,
                                                     validation_labels, validation_spectra,
                                                     num_pixel=training_spectra.shape[1],
                                                     num_neurons=num_neurons, num_steps=num_steps,
                                                     num_features=num_features, mask_size=mask_size,
                                                     learning_rate=learning_rate)
nn_default_file.rename(NN_output_file)

model_params = dict(dim_in=training_labels.shape[1],
                    num_neurons=num_neurons,
                    num_features=num_features,
                    mask_size=mask_size,
                    num_pixel=training_spectra.shape[1])
np.savez(model_param_output_file, **model_params)

print('Training Complete!')
