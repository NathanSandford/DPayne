import numpy as np
import matplotlib.pyplot as plt
from The_Payne.spectral_model import get_spectrum_from_neural_net


def read_in_neural_network(nn_file):
    '''
    Adjusted from The_Payne
    :param nn_file:
    :return:
    '''
    tmp = np.load(nn_file)
    w_array_0 = tmp["w_array_0"]
    w_array_1 = tmp["w_array_1"]
    w_array_2 = tmp["w_array_2"]
    b_array_0 = tmp["b_array_0"]
    b_array_1 = tmp["b_array_1"]
    b_array_2 = tmp["b_array_2"]
    x_min = tmp["x_min"]
    x_max = tmp["x_max"]
    NN_coeffs = (w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max)
    tmp.close()
    return NN_coeffs


def get_spectrum_from_nn(labels, nn_coeffs, nhidden, scaled=False, method='pix_by_pix'):
    """

    :param labels:
    :param nn_coeffs:
    :param nhidden:
    :param scaled:
    :param method:
    :return:
    """
    x_min = nn_coeffs[-2]
    x_max = nn_coeffs[-1]
    if scaled:
        scaled_labels = labels
    else:
        scaled_labels = (labels - x_min) / (x_max - x_min) - 0.5

    if nhidden == 1:
        w0, w1, b0, b1, x_min, x_max = nn_coeffs
        if method == 'pix_by_pix':
            inside = np.einsum('ijk,k->ij', w0, scaled_labels) + b0
            outside = np.einsum('ij,ij->i', w1, sigmoid(inside)) + b1
        elif method == 'all_pix':
            inside = np.einsum('ij,j->i', w0, scaled_labels) + b0
            outside = np.einsum('ij,j->i', w1, sigmoid(inside)) + b1
        else:
            raise ValueError(f'method {method} not understood')
    elif nhidden == 2:
        w0, w1, w2, b0, b1, b2, x_min, x_max = nn_coeffs
        if method == 'pix_by_pix':
            inside = sigmoid(np.einsum('ijk,k->ij', w0, scaled_labels) + b0)
            middle = sigmoid(np.einsum('ij->i', w1 * inside) + b1)
            outside = w2 * middle + b2
        elif method == 'all_pix':
            inside = np.einsum('ij,j->i', w0, scaled_labels) + b0
            middle = np.einsum('ij,j->i', w1, sigmoid(inside)) + b1
            outside = np.einsum('ij,j->i', w2, sigmoid(middle)) + b2
        else:
            raise ValueError(f'method {method} not understood')
    else:
        raise ValueError(f'nhidden must be 1 or 2, not {nhidden}')
    return outside


def get_spectrum_from_nn_ttsafe(labels, nn_coeffs, nhidden, scaled=False):
    """

    :param labels:
    :param nn_coeffs:
    :param nhidden:
    :param scaled:
    :return:
    """
    x_min = nn_coeffs[-2]
    x_max = nn_coeffs[-1]
    if scaled:
        scaled_labels = labels
    else:
        scaled_labels = (labels - x_min) / (x_max - x_min) - 0.5

    if nhidden == 1:
        w0, w1, b0, b1, x_min, x_max = nn_coeffs
        inside = sigmoid(w0.dot(scaled_labels) + b0)
        outside = w1.dot(inside) + b1
    elif nhidden == 2:
        w0, w1, w2, b0, b1, b2, x_min, x_max = nn_coeffs
        inside = sigmoid(w0.dot(scaled_labels) + b0)
        middle = sigmoid(w1.dot(inside) + b1)
        outside = w2.dot(middle) + b2
    else:
        raise ValueError(f'nhidden must be 1 or 2, not {nhidden}')
    return outside


def get_gradient_from_nn(labels, nn_coeffs, nhidden, dX, scaled=False, tensor_safe=False, method='pix_by_pix'):
    if tensor_safe:
        spec_p = get_spectrum_from_nn_ttsafe(labels + dX, nn_coeffs, nhidden, scaled=scaled)
        spec_m = get_spectrum_from_nn_ttsafe(labels - dX, nn_coeffs, nhidden, scaled=scaled)
    else:
        spec_p = get_spectrum_from_nn(labels + dX, nn_coeffs, nhidden, scaled=scaled, method=method)
        spec_m = get_spectrum_from_nn(labels - dX, nn_coeffs, nhidden, scaled=scaled, method=method)
    grad = (spec_p - spec_m) / (2 * np.linalg.norm(dX))
    return grad


class NeuralNet:
    def __init__(self, nn_file):
        self.NN_Coeffs = read_in_neural_network(nn_file)
        self.nn_file = nn_file
        self.num_neurons = self.NN_Coeffs[0].shape[0]

    def spectrum(self, labels, scaled=False):
        if scaled:
            if labels[0] > 1000:
                print('WARNING: Teff > 1000, are you sure your labels are scaled?')
        else:
            labels = self.scale_labels(labels)
        spec = get_spectrum_from_neural_net(labels, self.NN_Coeffs)
        return spec

    def scale_labels(self, labels):
        return scale_labels(labels, self.NN_Coeffs[-2], self.NN_Coeffs[-1])

    def rescale_labels(self, scaled_labels):
        return rescale_labels(scaled_labels, self.NN_Coeffs[-2], self.NN_Coeffs[-1])

    def plot_loss(self, savefig=False):
        temp = np.load(self.nn_file)  # the output array also stores the training and validation loss
        training_loss = temp["training_loss"]
        validation_loss = temp["validation_loss"]
        plt.figure(figsize=(15, 5))
        plt.plot(np.arange(training_loss.size) * 1000, training_loss, 'k', lw=0.5, label='Training set')
        plt.plot(np.arange(training_loss.size) * 1000, validation_loss, 'r', lw=0.5, label='Validation set')
        plt.legend(loc='best', frameon=False, fontsize=18)
        plt.xlabel("Step", size=20)
        plt.ylabel("Loss", size=20)
        if savefig:
            plt.savefig(f'NN_data/{savefig}.png')
        plt.show()

    def fit_spec(self, spectrum, p0=None, num_p0=1, tol=5e-4):
        from .fitting import fit_normalized_spec
        return fit_normalized_spec(spectrum, self, p0=p0, num_p0=num_p0, tol=tol)


