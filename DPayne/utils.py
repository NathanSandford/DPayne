import yaml
import numpy as np

from pathlib import Path
import torch
from .neural_networks import PaynePerceptron, PayneResnet


def split_data(spectra, labels, labels_to_train_on, frac_train=0.75, randomize=False, random_state=None):
    n_train = int(frac_train * spectra.shape[1])
    if randomize:
        spectra.columns = np.arange(spectra.shape[1])
        labels.columns = np.arange(labels.shape[1])
        training_sample = spectra.sample(n_train, axis=1, random_state=random_state).columns
        validation_sample = list(set(training_sample) ^ set(labels.columns))
        training_spectra = spectra[training_sample].values.T
        training_labels = labels[training_sample].loc[labels_to_train_on].values.T
        validation_spectra = spectra[validation_sample].values.T
        validation_labels = labels[validation_sample].loc[labels_to_train_on].values.T
    else:
        training_spectra = spectra.values.T[:n_train, :]
        training_labels = labels.loc[labels_to_train_on].values.T[:n_train, :]
        validation_spectra = spectra.values.T[n_train:, :]
        validation_labels = labels.loc[labels_to_train_on].values.T[n_train:, :]
    return training_labels, training_spectra, validation_labels, validation_spectra


def scale_labels(labels, x_min, x_max):
    return (labels - x_min) / (x_max - x_min) - 0.5


def rescale_labels(scaled_labels, x_min, x_max):
    return (scaled_labels + 0.5) * (x_max - x_min) + x_min


def doppler_shift(wave, spec, RV):
    """

    :param wave: 
    :param spec: 
    :param RV: 
    :return: 
    """
    c = 2.99792458e5  # km/s
    doppler_factor = np.sqrt((1 - RV/c)/(1 + RV/c))
    new_wavelength = wave * doppler_factor
    return np.interp(new_wavelength, wave, spec)


def generate_wavelength_template(start_wavelength: float, end_wavelength: float,
                                 resolution: float, truncate: bool = False):
    """

    :param start_wavelength:
    :param end_wavelength:
    :param resolution:
    :param truncate:
    :return:
    """
    wavelength_template = [start_wavelength]
    wavelength_now = start_wavelength
    while wavelength_now < end_wavelength:
        wavelength_now += wavelength_now / resolution
        wavelength_template.append(wavelength_now)
    wavelength_template = np.array(wavelength_template)
    if truncate:
        wavelength_template = wavelength_template[:-1]
    return wavelength_template
