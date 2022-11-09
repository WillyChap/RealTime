###################
# Referenced code #
###################
#
# Author: Weiming Hu, Will Chapman
# Source: https://github.com/WillyChap/ARML_Probabilistic/blob/54dc5b06b84f0a023e394ae24e9e0e1ea49301e1/Coastal_Points/python_scripts/coms.py
#
# - CRPS from Normal Distribution: https://projects.ecoforecast.org/neon4cast-docs/evaluation.html#crps-from-the-normal-distribution
#

import os
import sys

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from distutils import util

from tensorflow import math as tfm

additional_path = os.path.expanduser('~/github/2021_ProbPrecip_Hu/Shared/')

if additional_path not in sys.path:
    sys.path.append(additional_path)
    
# import helpers

MASK_INDICES = helpers.get_mask(return_indices=True)
MASK_HEIGHT = 228
MASK_WIDTH = 211
CSGD_CHANNELS = 3
CRPS_CHANNELS = 2


def apply_mask(y_true, y_pred, num_channels):
    y_true = tf.reshape(y_true, (-1, MASK_HEIGHT * MASK_WIDTH, 1, 1))
    y_true = tf.gather(y_true, MASK_INDICES, axis=1)

    y_pred = tf.reshape(y_pred, (-1, MASK_HEIGHT * MASK_WIDTH, 1, num_channels))
    y_pred = tf.gather(y_pred, MASK_INDICES, axis=1)

    return y_true, y_pred


def metric_crps(mu, sigma, truth):
    """
    http://cran.nexr.com/web/packages/scoringRules/vignettes/crpsformulas.html
    """
    
    # The following three variables are just for convenience
    loc = (truth - mu) / sigma
    phi = 1.0 / np.sqrt(2.0 * np.pi) * K.exp(-K.square(loc) / 2.0)
    Phi = 0.5 * (1.0 + tfm.erf(loc / np.sqrt(2.0)))
    
    # First we will compute the crps for each input/target pair
    crps_arr = sigma * (loc * (2. * Phi - 1.) + 2 * phi - 1. / np.sqrt(np.pi))
    
    return crps_arr


def crps_cost_function(y_true, y_pred, reduce_sum=True):
    """
    Compute the CRPS cost function for a normal distribution defined by
    the mean and standard deviation.
    Code from Will Chapman, inspired by Kai Polsterer (HITS).
    
    Args:
        y_true: True values
        y_pred: Tensor containing predictions: [mean, std]
    Returns:
        mean_crps: Scalar with mean CRPS over batch
    """

    # Split input
    mu = y_pred[:,:,:,0]
    sigma = K.abs(y_pred[:,:,:,1]) + 1e-5
    y_true = tf.squeeze(y_true, 3)
    
    # Calculate CRPS
    crps_arr = metric_crps(mu, sigma, y_true)
    
    if reduce_sum:
        return K.mean(crps_arr)
    else:
        return crps_arr
    
    
def csgd_cost_function(y_true, y_pred, reduce_sum=True, min_sigma_ratio=None):
    """
    Code by Vaghef Ghazvinian
    
    This function sets the loss function: crps for censored shifted gamma distribution.
    see the closed form expression in Scheuerer and Hamill 2015, Ghazvinian et al. 2021.
    
        - https://doi.org/10.1016/j.advwatres.2021.103907
        - https://doi.org/10.1175/MWR-D-15-0061.1
        
    shift parameter is also optimized here, to avoid shift become positive, I first square it
    then use negative of square root. log(mu) and log(sigma) are additional outputs. 
    
    return: mean value of crps over batch 
    """
    
    if 'CSGD_MIN_SIGMA_RATIO' in os.environ:
        min_sigma_ratio = float(os.environ['CSGD_MIN_SIGMA_RATIO'])
        
    assert min_sigma_ratio is not None
    
    obs = tf.squeeze(y_true, 3)
    
    shift = -(K.square(y_pred[:, :, :, 0]) + 1e-2)
    
    mu = -shift + K.square(y_pred[:, :, :, 1]) + 1e-2

    sigma_offset = K.square(y_pred[:, :, :, 2])
    sigma = sigma_offset + mu * min_sigma_ratio
    # sigma = tf.clip_by_value(sigma, 0.0, 1.0e3)
    
    # print('mu: {}; sigma: {}; shift: {}'.format(mu, sigma, shift))
    
    shape = K.square(mu / sigma)
    scale = (K.square(sigma)) / mu
    
    # print('shape: {}; scale: {}; shift: {}'.format(shape, scale, shift))
    
    # First term in Eq. (5)
    y_bar = (obs - shift) / scale
    F_k_y = tf.math.igamma(shape, 1. * y_bar)
    
    c1 = y_bar * (2. * F_k_y - 1.)
    
    # Second term in Eq. (5)
    # B_05_kp05 = tf.vectorized_map(lambda x: K.exp(tf.math.lbeta([0.5, x])), shape + 0.5)
    B_05_kp05 = K.exp(tf.math.lbeta(tf.stack([tf.fill(tf.shape(shape), 0.5), shape + 0.5],
                                             axis=len(shape.shape))))
    
    c_bar = (-1 * shift) / scale
    F_2k_2c = tf.math.igamma(2. * shape, 1. * 2. * c_bar)
    
    c4 = (shape / np.pi) * B_05_kp05 * (1. - F_2k_2c)
    
    # Third term in Eq. (5)
    F_k_c = tf.math.igamma(shape, 1. * c_bar)
    F_kp1_y = tf.math.igamma(shape+1., 1. * y_bar)
    F_kp1_c = tf.math.igamma(shape+1., 1. * c_bar)
    
    c2 = shape * (2. * F_kp1_y - 1. + K.square(F_k_c) - 2. * F_kp1_c * F_k_c)
    
    # Fourth term in Eq. (5)
    c3 = c_bar * K.square(F_k_c)

    # print('c1: {} c2: {} c3: {} c4: {}'.format(c1, c2, c3, c4))
    
    crps = c1 - c2 - c3 - c4        
    
    CRPS = crps * scale
    
    CRPS += sigma_offset * 0.01
    
    if reduce_sum:
        return K.mean(CRPS)
    else:
        return CRPS
    

def ign_cost_function(y_true, y_pred, reduce_sum=True, use_tfd=False):
    if use_tfd:
        from tensorflow_probability import distributions as tfd
    
    ##Written by Vaghef Ghazvinian, mghazvinian@ucsd.edu
    ##Inspired by the work of Alex Cannon https://doi.org/10.1175/2008JHM960.1 for downscaling.
    ##The loss basically models precipitation data in a two part scheme, a mass probability
    ##combined to a positive distribution (gamma here) for the magnitude of precipitation.
    ##This loss has not been used for precipitation post processing before as far as I know. My limited evaluations
    ##but show this is more robust than ANN-CSGD but for the efficacy in terms of forecast skill we need to test this 
    ##as it uses negative log likelihood instead of CRPS.

    logit_pop = y_pred[:, :, :, 0]
    pop = K.square(logit_pop) / (1 + K.square(logit_pop))
    pop = tf.clip_by_value(pop, 1e-5, 1.0 - 1e-5)

    log_mu = y_pred[:, :, :, 1]
    mu = K.square(log_mu) + 1e-3

    log_sigma= y_pred[:, :, :, 2]
    sigma = K.square(log_sigma) + 1e-3
        
    obs = tf.squeeze(y_true, 3)
    positive_obs = obs > 0.
    zero_obs = obs == 0.
    
    shape = K.square(mu / sigma)
    scale = (K.square(sigma)) / mu
    rate = 1. / scale
      
    density = 1. - pop[zero_obs]
    loss_zero = -tf.math.log(density)
    
    shape_pos = tf.convert_to_tensor(shape[positive_obs])
    rate_pos = tf.convert_to_tensor(rate[positive_obs])
    
    if use_tfd:
        # This block uses tensorflow probability that requires higher versions of tensorflow (2.7.0)
        dist = tfd.Gamma(concentration=shape_pos, rate=rate_pos)
        obs_pos = tf.convert_to_tensor(obs[positive_obs])
        prob = pop[positive_obs] * dist.prob(obs_pos)
        loss_pos = -tf.math.log(prob + 1e-5)
        loss = tf.concat([loss_zero, loss_pos], axis=0)
        
    else:
        obs_pos=tf.convert_to_tensor(obs[positive_obs])
        shape_pos = tf.convert_to_tensor(shape[positive_obs])
        rate_pos = tf.convert_to_tensor(rate[positive_obs])
        
        log_unnormalized_prob = tf.math.xlogy(shape_pos - 1., obs_pos) - tf.math.multiply(rate_pos, obs_pos)
        log_normalization = (tf.math.lgamma(shape_pos) -tf.math.multiply(shape_pos, tf.math.log(rate_pos)))
        log_prob_gamma = log_unnormalized_prob - log_normalization
        log_prob_gamma = tf.clip_by_value(log_prob_gamma, -1.e5, 0.)
        
        prob = pop[positive_obs] * K.exp(log_prob_gamma)
        
        loss_pos = -tf.math.log(prob + 1e-5)
        loss=tf.concat([loss_zero, loss_pos], axis=0)
    
    if reduce_sum:
        return K.mean(loss)
    else:
        return loss
    
    
def ign_cost_function_old(y_true, y_pred, reduce_sum=True, use_tfd=False):
    """
    This function sets the loss function: negative log-likelihood for zero-inflated gamma pdf
    loss=-log[f(y)]
    f[y=0] = 1-pop
    f[y>0] = pop[x>0]*dgamma(y[y>0], scale=scale[y>0], shape=shape[y>0])
    outputs are set to logit(pop), log(mu) and log(sigma)to set a limit the output for pop
    in range (0,1) and mu and sigma to be positive. This is like using link functions in glm or gamlss.
    return: mean value of ign score over batch 
    
    concentration is the shape of gamma
    rate= 1. / scale
    y_pred represents NN outputs 
    """
    
    if use_tfd:
        from tensorflow_probability import distributions as tfd
    
    ##Written by Vaghef Ghazvinian, mghazvinian@ucsd.edu
    ##Inspired by the work of Alex Cannon https://doi.org/10.1175/2008JHM960.1 for downscaling.
    ##The loss basically models precipitation data in a two part scheme, a mass probability
    ##combined to a positive distribution (gamma here) for the magnitude of precipitation.
    ##This loss has not been used for precipitation post processing before as far as I know. My limited evaluations
    ##but show this is more robust than ANN-CSGD but for the efficacy in terms of forecast skill we need to test this 
    ##as it uses negative log likelihood instead of CRPS.

    logit_pop = y_pred[:, :, :, 0]
    pop = K.square(logit_pop) / (1 + K.square(logit_pop))
    pop = tf.clip_by_value(pop, 1e-5, 1.0 - 1e-5)

    log_mu = y_pred[:, :, :, 1]
    mu = K.square(log_mu) + 1e-3

    log_sigma= y_pred[:, :, :, 2]
    sigma = K.square(log_sigma) + 1e-3
        
    obs = tf.squeeze(y_true, 3)
    positive_obs = obs > 0.
    zero_obs = obs == 0.
    
    # sigma < K.abs(obs[positive_obs] - mu[positive_obs])
       
    shape = K.square(mu / sigma)
    scale = (K.square(sigma)) / mu
    rate = 1. / scale
      
    density = 1. - pop[zero_obs]
    loss_zero = -tf.math.log(density)
    
    shape_pos = tf.convert_to_tensor(shape[positive_obs])
    rate_pos = tf.convert_to_tensor(rate[positive_obs])
    
    if use_tfd:
        # This block uses tensorflow probability that requires higher versions of tensorflow (2.7.0)
        dist = tfd.Gamma(concentration=shape_pos, rate=rate_pos)
        obs_pos = tf.convert_to_tensor(obs[positive_obs])
        loss_pos=-dist.log_prob(obs_pos)-tf.math.log(pop[positive_obs])
        loss = tf.concat([loss_zero, loss_pos], axis=0)
        
    else:
        obs_pos=tf.convert_to_tensor(obs[positive_obs])
        shape_pos = tf.convert_to_tensor(shape[positive_obs])
        rate_pos = tf.convert_to_tensor(rate[positive_obs])
        
        log_unnormalized_prob = tf.math.xlogy(shape_pos - 1., obs_pos) - tf.math.multiply(rate_pos, obs_pos)
        log_normalization = (tf.math.lgamma(shape_pos) -tf.math.multiply(shape_pos, tf.math.log(rate_pos)))
        loss_pos =  log_normalization - log_unnormalized_prob -  tf.math.log(pop[positive_obs])
        loss=tf.concat([loss_zero, loss_pos], axis=0)
    
    if reduce_sum:
        return K.mean(loss)
    else:
        return loss


def to_cce_categories(arr):
    # Input arr has shape [n_samples, h, w, 1]
    # Output arr has shape [n_samples, h, w, n_categories] with one-hot encoding
    
    # TODO
    return np.random.rand(0, 1, 2, 3)
    
    
def cce_cost_function(y_true, y_pred, reduce_sum=True):
    # y_true has shape [n_samples, h, w, 1]
    # y_pred has shape [n_samples, h, w, n_categories]
    
    # Convert y_true to one-hot encodings with predefined category boundaries    
    y_true = to_cce_categories(y_true)
    
    loss = y_true * y_pred
    loss = K.sum(loss, axis=-1)
    loss = K.log(loss)
    
    if reduce_sum:
        return -K.mean(loss)
    else:
        return loss
    

def get_cost_function(name):
    
    if name == 'crps':
        return crps_cost_function
    elif name == 'masked_crps':
        return lambda y_true, y_pred: crps_cost_function(*apply_mask(y_true, y_pred, CRPS_CHANNELS))
    
    elif name == 'csgd':
        return csgd_cost_function
    elif name == 'masked_csgd':
        return lambda y_true, y_pred: csgd_cost_function(*apply_mask(y_true, y_pred, CSGD_CHANNELS))
    
    elif name == 'ign':
        return ign_cost_function
    elif name == 'masked_ign':
        return lambda y_true, y_pred: ign_cost_function(*apply_mask(y_true, y_pred, CSGD_CHANNELS))
    
    elif name == 'mse':
        return 'mse'
    
    elif name == 'cce':
        raise Exception('Must use masked categorical cross entropy loss (masked_cce)')
    elif name == 'masked_cce':
        return cce_cost_function
    
    else:
        raise Exception('Unrecognized cost function name: {}'.format(name))
