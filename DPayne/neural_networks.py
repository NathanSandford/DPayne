"""
This file is used to train the neural networks that predict the spectrum
given any set of stellar labels (stellar parameters + elemental abundances).

Note that, the approach here is different from Ting+19. Instead of
training individual small networks for each pixel separately, we train a single
large network for all pixels simultaneously.

We found that for most cases, a simple multilayer perceptron model (only dense layers)
works sufficiently well. But here we also provide the option to run a large deconvolutional
ResNet. In the case where the parameter space spanned is large, ResNet can provide better
emulation.

The default training set are synthetic spectra the Kurucz models and have been
convolved to the appropriate R (~22500 for APOGEE) with the APOGEE LSF.
"""

from pathlib import Path
import yaml
import numpy as np
import torch
from torch.autograd import Variable
from . import radam
from . import torch2theano
from .utils import scale_labels


class Model:
    def __init__(self, model_par):
        self.parameters = model_par
        self.name = model_par['name']
        self.arch_type = model_par['arch_type']
        self.num_pixel = model_par['num_pixel']
        self.labels = model_par['labels']
        self.dim_in = len(self.labels)
        self.num_neurons = model_par['num_neurons']
        if self.arch_type == 'perceptron':
            self.nn = PaynePerceptron(dim_in=self.dim_in,
                                      num_neurons=self.num_neurons,
                                      num_pixel=self.num_pixel)
        elif self.arch_type == 'resnet':
            self.pix_per_channel = model_par['pix_per_channel']
            self.kernel_size = model_par['kernel_size']
            self.stride = model_par['stride']
            self.padding = model_par['padding']
            self.nn = PayneResnet(dim_in=self.dim_in,
                                  num_pixel=self.num_pixel,
                                  num_neurons=self.num_neurons,
                                  pix_per_channel=self.pix_per_channel,
                                  kernel_size=self.kernel_size,
                                  stride=self.stride,
                                  padding=self.padding)

        self.x_min = None
        self.x_max = None
        self.training_loss = None
        self.validation_loss = None
        self.nn_tt = None

    def load_model(self, model_file):
        state_dict = torch.load(model_file)
        self.nn.load_state_dict(state_dict)
        self.nn.eval()

    def load_scaling(self, scaling_file):
        with np.load(scaling_file) as tmp:
            self.x_min = tmp['x_min']
            self.x_max = tmp['x_max']

    def load_loss(self, loss_file):
        with np.load(loss_file) as tmp:
            self.training_loss = tmp['training_loss']
            self.validation_loss = tmp['validation_loss']

    def theano_wrap(self):
        self.nn_tt = torch2theano.pytorch_wrapper(self.nn, dtype=torch.FloatTensor)

    def spec(self, x, scaled=True):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x).type(torch.FloatTensor)
        elif type(x) is list:
            x = torch.FloatTensor(x)
        if x.dim() == 1:
            x = x[None, :]
        if not scaled:
            x = scale_labels(x, self.x_min, self.x_max)
        return self.nn(x)

    def spec_tt(self, x, scaled=True):
        if not scaled:
            x = scale_labels(x, self.x_min, self.x_max)
        return self.nn_tt(x).eval()


# ===================================================================================================
# simple multi-layer perceptron model
class PaynePerceptron(torch.nn.Module):
    def __init__(self, dim_in, num_pixel, num_neurons):
        super(PaynePerceptron, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Linear(dim_in, num_neurons),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(num_neurons, num_neurons),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(num_neurons, num_pixel),
        )

    def forward(self, x):
        return self.features(x)

    def load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)


# ---------------------------------------------------------------------------------------------------
# resnet models
class PayneResnet(torch.nn.Module):
    def __init__(self, dim_in, num_pixel, num_neurons, pix_per_channel, kernel_size, stride, padding):
        super(PayneResnet, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Linear(dim_in, num_neurons),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(num_neurons, num_neurons),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(num_neurons, 64*pix_per_channel),
        )

        self.deconv1 = torch.nn.ConvTranspose1d(64, 64, kernel_size, stride=stride, padding=padding)
        self.deconv2 = torch.nn.ConvTranspose1d(64, 64, kernel_size, stride=stride, padding=padding)
        self.deconv3 = torch.nn.ConvTranspose1d(64, 64, kernel_size, stride=stride, padding=padding)
        self.deconv4 = torch.nn.ConvTranspose1d(64, 64, kernel_size, stride=stride, padding=padding)
        self.deconv5 = torch.nn.ConvTranspose1d(64, 64, kernel_size, stride=stride, padding=padding)
        self.deconv6 = torch.nn.ConvTranspose1d(64, 32, kernel_size, stride=stride, padding=padding)
        self.deconv7 = torch.nn.ConvTranspose1d(32,  1, kernel_size, stride=stride, padding=padding)

        self.deconv2b = torch.nn.ConvTranspose1d(64, 64, 1, stride=stride)
        self.deconv3b = torch.nn.ConvTranspose1d(64, 64, 1, stride=stride)
        self.deconv4b = torch.nn.ConvTranspose1d(64, 64, 1, stride=stride)
        self.deconv5b = torch.nn.ConvTranspose1d(64, 64, 1, stride=stride)
        self.deconv6b = torch.nn.ConvTranspose1d(64, 32, 1, stride=stride)

        self.relu2 = torch.nn.LeakyReLU()
        self.relu3 = torch.nn.LeakyReLU()
        self.relu4 = torch.nn.LeakyReLU()
        self.relu5 = torch.nn.LeakyReLU()
        self.relu6 = torch.nn.LeakyReLU()

        self.num_pixel = num_pixel
        self.pix_per_channel = pix_per_channel

    def forward(self, x):
        x = self.features(x)[:, None, :]
        x = x.view(x.shape[0], 64, self.pix_per_channel)
        x1 = self.deconv1(x)

        x2 = self.deconv2(x1)
        x2 += self.deconv2b(x1)
        x2 = self.relu2(x2)

        x3 = self.deconv3(x2)
        x3 += self.deconv3b(x2)
        x3 = self.relu2(x3)

        x4 = self.deconv4(x3)
        x4 += self.deconv4b(x3)
        x4 = self.relu2(x4)

        x5 = self.deconv5(x4)
        x5 += self.deconv5b(x4)
        x5 = self.relu2(x5)

        x6 = self.deconv6(x5)
        x6 += self.deconv6b(x5)
        x6 = self.relu2(x6)

        x7 = self.deconv7(x6)[:, 0, : self.num_pixel]
        return x7

    def load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)


# ===================================================================================================
# train neural networks
def train_nn(
    training_labels,
    training_spectra,
    validation_labels,
    validation_spectra,
    num_pixel=7214,
    num_neurons=300,
    num_steps=1e5,
    learning_rate=1e-4,
    batch_size=512,
    pix_per_channel=5,
    kernel_size=11,
    stride=3,
    padding=5,
    arch_type="perceptron",
    nn_dir="./neural_nets",
    model_name="NN",
    continue_from_model=False,
):

    """
    Training neural networks to emulate spectral models

    training_labels has the dimension of [# training spectra, # stellar labels]
    training_spectra has the dimension of [# training spectra, # wavelength pixels]

    The validation set is used to independently evaluate how well the neural networks
    are emulating the spectra. If the networks overfit the spectral variation, while
    the loss function will continue to improve for the training set, but the validation
    set should show a worsen loss function.

    The training is designed in a way that it always returns the best neural networks
    before the networks start to overfit (gauged by the validation set).

    num_steps = how many steps to train until convergence.
    1e5 is good for the specific NN architecture and learning I used by default,
    but bigger networks take more steps, and decreasing the learning rate will
    also change this. You can get a sense of how many steps are needed for a new
    NN architecture by plotting the loss function evaluated on both the training set
    and a validation set as a function of step number. It should plateau once the NN
    has converged.

    learning_rate = step size to take for gradient descent
    This is also tunable, but 1e-4 seems to work well for most use cases. Again,
    diagnose with a validation set if you change this.

    pix_per_channel is the number of features before the deconvolutional layers; it only
    applies if ResNet is used. For the simple multi-layer perceptron model, this parameter
    is not used. We truncate the predicted model if the output number of pixels is
    larger than what is needed. In the current default model, the output is ~8500 pixels
    in the case where the number of pixels is > 8500, increase the number of features, and
    tweak the ResNet model accordingly

    batch_size = the batch size for training the neural networks during the stochastic
    gradient descent. A larger batch_size reduces the stochasticity, but it might also
    risk to stuck in a local minimum

    returns:
        training loss and validation loss
    """

    # Set paths and file names
    nn_dir = Path(nn_dir)
    scaling_file = nn_dir.joinpath(f'{model_name}_scaling.npz')
    model_file = nn_dir.joinpath(f'{model_name}_model.pt')
    loss_file = nn_dir.joinpath(f'{model_name}_loss.npz')

    # run on cuda
    dtype = torch.cuda.FloatTensor
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

    # scale the labels, optimizing neural networks is easier if the labels are more normalized
    x_max = np.max(training_labels, axis=0)
    x_min = np.min(training_labels, axis=0)
    x = scale_labels(training_labels, x_min, x_max)
    x_valid = scale_labels(validation_labels, x_min, x_max)

    # save scaling relation
    np.savez(scaling_file, x_min=x_min, x_max=x_max)

    # dimension of the input
    dim_in = x.shape[1]

    # --------------------------------------------------------------------------------------------
    # assume L2 loss
    loss_fn = torch.nn.L1Loss(reduction="mean")

    # make pytorch variables
    x = Variable(torch.from_numpy(x)).type(dtype)
    y = Variable(torch.from_numpy(training_spectra), requires_grad=False).type(dtype)
    x_valid = Variable(torch.from_numpy(x_valid)).type(dtype)
    y_valid = Variable(torch.from_numpy(validation_spectra), requires_grad=False).type(dtype)

    # initiate Payne and optimizer
    if arch_type == "perceptron":
        model = PaynePerceptron(dim_in, num_pixel, num_neurons)
    elif arch_type == "resnet":
        model = PayneResnet(dim_in, num_pixel, num_neurons, pix_per_channel, kernel_size, stride, padding)
    if continue_from_model:
        model.load(model_file)

    model.cuda()
    model.train()

    # we adopt rectified Adam for the optimization
    optimizer = radam.RAdam(
        [p for p in model.parameters() if p.requires_grad], lr=learning_rate
    )

    # --------------------------------------------------------------------------------------------
    # break into batches
    nsamples = x.shape[0]
    nbatches = nsamples // batch_size

    nsamples_valid = x_valid.shape[0]
    nbatches_valid = nsamples_valid // batch_size

    # initiate counter
    if continue_from_model:
        tmp = np.load(loss_file)
        training_loss = list(tmp["training_loss"])
        validation_loss = list(tmp["validation_loss"])
        current_loss = training_loss[-1]
        epoch = len(training_loss)
    else:
        current_loss = np.inf
        training_loss = []
        validation_loss = []
        epoch = 0

    # -------------------------------------------------------------------------------------------------------
    # train the network
    for _e in range(int(num_steps) - epoch):
        e = _e + epoch

        # randomly permute the data
        perm = torch.randperm(nsamples)
        perm = perm.cuda()

        # for each batch, calculate the gradient with respect to the loss
        for i in range(nbatches):
            idx = perm[i * batch_size: (i + 1) * batch_size]
            y_pred = model(x[idx])

            loss = loss_fn(y_pred, y[idx]) * 1e4
            optimizer.zero_grad()
            loss.backward(retain_graph=False)
            optimizer.step()

        # the average loss
        if e % 100 == 0:

            # randomly permute the data
            perm_valid = torch.randperm(nsamples_valid)
            perm_valid = perm_valid.cuda()
            loss_valid = 0

            for j in range(nbatches_valid):
                idx = perm_valid[j * batch_size: (j + 1) * batch_size]
                y_pred_valid = model(x_valid[idx])
                loss_valid += loss_fn(y_pred_valid, y_valid[idx]) * 1e4
            loss_valid /= nbatches_valid

            print(f"iter {e}: training loss = {loss: .3f} validation loss = {loss_valid: .3f}")

            loss_data = loss.detach().data.item()
            loss_valid_data = loss_valid.detach().data.item()
            training_loss.append(loss_data)
            validation_loss.append(loss_valid_data)

            # record the weights and biases if the validation loss improves
            if loss_valid_data < current_loss:
                current_loss = loss_valid_data

                state_dict = model.state_dict()
                for k, v in state_dict.items():
                    state_dict[k] = v.cpu()
                torch.save(state_dict, model_file)

                np.savez(
                    loss_file,
                    training_loss=training_loss,
                    validation_loss=validation_loss,
                )

    # --------------------------------------------------------------------------------------------
    # save the final training loss
    np.savez(
        loss_file,
        training_loss=training_loss,
        validation_loss=validation_loss,
    )
    return


def load_trained_model(model_name, nn_dir, theano_wrap=False):
    modelpar_file = Path(nn_dir).joinpath(f'{model_name}_par.yml')
    model_file = Path(nn_dir).joinpath(f'{model_name}_model.pt')
    scaling_file = Path(nn_dir).joinpath(f'{model_name}_scaling.npz')
    loss_file = Path(nn_dir).joinpath(f'{model_name}_loss.npz')
    with open(modelpar_file, 'r') as infile:
        model_par = yaml.safe_load(infile)
    model = Model(model_par)
    model.load_model(model_file)
    model.load_scaling(scaling_file)
    model.load_loss(loss_file)
    if theano_wrap:
        model.theano_wrap()
    return model
