import sys
sys.path.append('/global/scratch/nathan_sandford/DPayne/')
import pandas as pd
from DPayne.utils import split_data
from The_Payne import training

input_file = sys.argv[1]
output_file = sys.argv[2]

# Sample Settings
frac_train = 0.75
randomize = True
random_state = 3457
element_labels = ['Fe', 'Ca', 'Ni', 'Si', 'Ti', 'Co', 'Mg']
labels_to_train_on = ['Teff', 'logg', 'v_micro'] + element_labels

# NN Settings
num_neurons = 300
num_steps = 1e5
learning_rate = 0.001

spectra = pd.read_hdf(input_file, 'spectra')
labels = pd.read_hdf(input_file, 'labels')
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
                                                     learning_rate=learning_rate)

print('Training Complete!')
