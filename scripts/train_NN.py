import sys
sys.path.append('/global/scratch/nathan_sandford/DPayne/')
import argparse
from pathlib import Path
import yaml
import pandas as pd
from DPayne.utils import split_data
from DPayne import neural_networks

description = \
    '''
    Code trains "The Payne" on synthetic spectra.
    '''

'''
Defaults
'''
base_dir = Path('/global/scratch/nathan_sandford/DPayne')
spec_dir = base_dir.joinpath('training_data')
nn_dir = base_dir.joinpath('neural_nets')

# Sample Settings
elements_to_train_on = ['Fe', 'Ca', 'Ni', 'Si', 'Ti', 'Co', 'Mg']
other_to_train_on = ['Teff', 'logg', 'v_micro']
frac_train = 0.75
random_state = 3457

# NN Settings
num_neurons = 300
num_steps = 1e5
learning_rate = 1e-4
batch_size = 512
pix_per_channel = 5
kernel_size = 11
stride = 3
padding = 5

'''
Parse Args
'''
parser = argparse.ArgumentParser(description=description)
parser.add_argument("model_name", help="Model name")
parser.add_argument("spec_file", help="File containing training spectra")
parser.add_argument("arch_type", help="NN architecture ('perceptron' or 'resnet')")

parser.add_argument("--base_dir", "-bdir", help=f"Base Directory (default: {base_dir.name})")
parser.add_argument("--spec_dir", "-sdir", help=f"Directory of training spec (default: {spec_dir.name})")
parser.add_argument("--nn_dir", "-ndir", help=f"Output directory for NN (default: {nn_dir.name})")


parser.add_argument("--elements_to_train_on", "-X", nargs='+',
                    help=f"Elements to train on (default: {' '.join(elements_to_train_on)})")
parser.add_argument("--other_to_train_on", "-oX", nargs='+',
                    help=f"Other labels to train on (default: {' '.join(other_to_train_on)})")
parser.add_argument("--frac_train", "-frac", type=float,
                    help=f"Fraction of spectra to train on (default: {frac_train})")
parser.add_argument("--random_state", "-rand", type=int,
                    help=f"Random state for sampling training spectra (default: {random_state})")

parser.add_argument("--num_neurons", "-Nn", type=int,
                    help=f"Number of neurons (default: {num_neurons})")
parser.add_argument("--num_steps", "-Ns", type=int,
                    help=f"Number of steps (default: {num_steps})")
parser.add_argument("--learning_rate", "-lr", type=int,
                    help=f"Learning rate (default: {learning_rate})")
parser.add_argument("--batch_size", "-bs", type=int,
                    help=f"Batch size (default: {batch_size})")
parser.add_argument("--pix_per_channel", "-ppc", type=int,
                    help=f"Number of pixels per channel (resnet only, default: {pix_per_channel})")
parser.add_argument("--kernel_size", "-ms", type=int,
                    help=f"Size of convolution kernel (resnet only, default: {kernel_size})")
parser.add_argument("--stride", "-s", type=int,
                    help=f"Stride for convolution kernel (resnet only, default: {stride})")
parser.add_argument("--padding", "-p", type=int,
                    help=f"Padding for convolution kernel (resnet only, default: {padding})")
parser.add_argument("--continue_from_model", '-cont', action="store_true", default=False,
                    help=f"Continue training existing model (default: False)")
args = parser.parse_args()

if args.base_dir:
    base_dir = Path(args.base_dir)
assert base_dir.is_dir(), f'Directory {base_dir} does not exist'

if args.spec_dir:
    spec_dir = base_dir.joinpath(args.spec_dir)
else:
    spec_dir = base_dir.joinpath(spec_dir.name)
assert spec_dir.is_dir(), f'Directory {spec_dir} does not exist'

if args.nn_dir:
    nn_dir = base_dir.joinpath(args.nn_dir)
else:
    nn_dir = base_dir.joinpath(nn_dir.name)
assert nn_dir.is_dir(), f'Directory {nn_dir} does not exist'

if args.elements_to_train_on:
    elements_to_train_on = args.elements_to_train_on
if args.other_to_train_on:
    other_to_train_on = args.other_to_train_on
if args.frac_train:
    frac_train = args.frac_train
if args.random_state:
    random_state = args.random_state

if args.num_neurons:
    num_neurons = args.num_neurons
if args.num_steps:
    num_steps = args.num_steps
if args.learning_rate:
    learning_rate = args.learning_rate
if args.batch_size:
    batch_size = args.batch_size
if args.pix_per_channel:
    pix_per_channel = args.pix_per_channel
if args.kernel_size:
    kernel_size = args.kernel_size
if args.stride:
    stride = args.stride
if args.padding:
    padding = args.padding

model_name = args.model_name
model_file = nn_dir.joinpath(f'{model_name}_model.pt')
modelpar_file = nn_dir.joinpath(f'{model_name}_par.npz')
scaling_file = nn_dir.joinpath(f'{model_name}_scaling.npz')
loss_file = nn_dir.joinpath(f'{model_name}_loss.npz')

spec_file = spec_dir.joinpath(args.spec_file)
assert spec_file.exists(), f'{spec_file} does not exist'

arch_type = args.arch_type
assert arch_type in ['perceptron', 'resnet'], f'arch_type must be "perceptron" or "resnet"'

continue_from_model = args.continue_from_model


'''
Prep Training Data
'''
wave = pd.read_hdf(spec_file, 'wavelength')
spectra = pd.read_hdf(spec_file, 'spectra')
labels = pd.read_hdf(spec_file, 'labels')
labels.loc[set(labels.index) ^ {'Teff', 'logg', 'v_micro', 'Fe'}] -= labels.loc['Fe'] # Scale by Iron
labels_to_train_on = other_to_train_on + elements_to_train_on
training_labels, training_spectra, validation_labels, validation_spectra \
    = split_data(spectra, labels, labels_to_train_on,
                 randomize=True, random_state=random_state)

'''
Init Model
'''
if arch_type == 'perceptron':
    model_par = dict(name=model_name,
                     arch_type=arch_type,
                     labels=labels_to_train_on,
                     num_pixel=training_spectra.shape[1],
                     num_neurons=num_neurons,
                     random_state=random_state,
                     )
elif arch_type == 'resnet':
    model_par = dict(name=model_name,
                     arch_type=arch_type,
                     labels=labels_to_train_on,
                     num_pixel=training_spectra.shape[1],
                     num_neurons=num_neurons,
                     pix_per_channel=pix_per_channel,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding=padding,
                     random_state=random_state,
                     )
with open(f'{model_name}_par.yml', 'wt') as outfile:
    yaml.dump(modelpar_file, outfile, sort_keys=False)


'''
Train The Payne
'''
print(f'Training on {training_spectra.shape[0]} spectra')
print(f'Validating w/ {validation_spectra.shape[0]} spectra')
print(f'Labels: {" ".join(labels_to_train_on)}')
neural_networks.train_nn(training_labels=training_labels,
                         training_spectra=training_spectra,
                         validation_labels=validation_labels,
                         validation_spectra=validation_spectra,
                         num_pixel=training_spectra.shape[1],
                         num_neurons=num_neurons,
                         num_steps=num_steps,
                         learning_rate=learning_rate,
                         batch_size=batch_size,
                         pix_per_channel=pix_per_channel,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         arch_type=arch_type,
                         nn_dir=nn_dir,
                         model_name=model_name,
                         continue_from_model=continue_from_model, )

print('Training Complete!')
