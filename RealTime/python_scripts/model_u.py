###################
# Referenced code #
###################
#
# Author: Weiming Hu, Will Chapman
# Source: https://github.com/WillyChap/ARML_Probabilistic/blob/54dc5b06b84f0a023e394ae24e9e0e1ea49301e1/Coastal_Points/python_scripts/Unet_b.py#L168
#

import os
import re
import yaml
import glob
import pickle
import inspect
# import utils_cost

import numpy as np
import tensorflow as tf
# import tensorflow_addons as tfa

import tensorflow.keras.layers as Layers
import tensorflow.keras.models as Models
import tensorflow.keras.initializers as Init

from importlib.machinery import SourceFileLoader


def initialization(kernel_size, prev_n_channels):
    return 'glorot_uniform'
    # return Init.RandomNormal(stddev=np.sqrt(2/((kernel_size ** 2) * prev_n_channels)))


def regularization(regularizer, regularizer_param):
    
    if regularizer is None:
        reg = None
        
    else:
        if regularizer == 'l2':
            reg = tf.keras.regularizers.l2(regularizer_param)
        elif regularizer == 'l1':
            reg = tf.keras.regularizers.l1(regularizer_param)
        else:
            raise Exception('Unknown regularizer: {}'.format(regularizer))
        
    return reg


def bottleneck(
    x, filters, kernel_size, num_convs, activation,
    regularizer, regularizer_param,
    batch_norm=False, use_dilation=False,
    conv2d_activation=True, last_activation=False,
    add_skip_layers=False):
    
    reg = regularization(regularizer, regularizer_param)
    
    skips = []
        
    for i in range(num_convs):

        x = Layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size, strides=1, padding='same', 
            kernel_initializer=initialization(kernel_size, x.shape[3]),
            kernel_regularizer=reg,
            bias_regularizer=reg,
            name = 'bottleneck_' + str(i),
            dilation_rate=2 ** i if use_dilation else 1)(x)
        
        if batch_norm:
            x = Layers.BatchNormalization()(x)
        
        if conv2d_activation:
            x = Layers.Activation(activation)(x)
        
        if add_skip_layers:
            skips.append(x)
    
    if add_skip_layers:
        x = Layers.add(skips)
    
    if last_activation:
        x = Layers.Activation(activation)(x)
        
    return x


def downsampling(
    x, level, filters, kernel_size, num_convs, activation,
    batch_norm, regularizer, regularizer_param,
    pool_size=2, pool_stride=2,
    connection_before_conv=False):
    
    reg = regularization(regularizer, regularizer_param)
    
    if connection_before_conv:
        skip = x
        
    for i in range(num_convs):

        x = Layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size, strides=1, padding='same',
            kernel_initializer=initialization(kernel_size, x.shape[3]),
            kernel_regularizer=reg,
            bias_regularizer=reg,
            name = 'downsampling_' + str(level) + '_conv_' + str(i))(x)
        
        if batch_norm:
            x = Layers.BatchNormalization(
                name='downsampling_' + str(level) + '_batchnorm_' + str(i))(x)
            
        x = Layers.Activation(
            activation, name='downsampling_' + str(level) + '_activation_' + str(i))(x)
    
    if not connection_before_conv:
        skip = x
        
    x = Layers.MaxPooling2D(pool_size=pool_size, strides=pool_stride, padding='same')(x)
    
    return x, skip


def upsampling(
    x, level, skip, filters, kernel_size, num_convs, activation,
    batch_norm, regularizer, regularizer_param,
    conv_transpose=True, upsampling_size=2, upsampling_strides=2):
    
    reg = regularization(regularizer, regularizer_param)    

    if conv_transpose:

        x = Layers.Conv2DTranspose(
            filters=filters,
            kernel_size = upsampling_size, strides=upsampling_strides,
            kernel_initializer = initialization(kernel_size, x.shape[3]),
            name = 'upsampling_' + str(level) + '_conv_trans_' + str(level))(x)
    else:
        x = Layers.UpSampling2D(
            (upsampling_size), name = 'upsampling_' + str(level) + '_ups_' + str(level))(x)
        
    x = Layers.Concatenate()([x, skip])
    
    for i in range(num_convs):

        x = Layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size, strides=1, padding='same',
            kernel_initializer=initialization(kernel_size, x.shape[3]),
            kernel_regularizer=reg,
            bias_regularizer=reg,
            name = 'upsampling_' + str(level) + '_conv_' + str(i))(x)
        
        if batch_norm:
            x = Layers.BatchNormalization(
                name = 'upsampling_' + str(level) + '_batchnorm_' + str(i))(x)
            
        x = Layers.Activation(
            activation, name = 'upsampling_' + str(level) + '_activation_' + str(i))(x)
        
    return x
    

def model_unet(
    img_shape, num_classes, num_levels, 
    num_layers, num_bottleneck, 
    cost_func, filter_size_start,
    batch_norm=False, kernel_size=3,
    use_dilation=True, conv2d_activation=True,
    last_activation=False, add_skip_layers=True,
    regularizer=None, regularizer_param=0.001,
    optimizer='adam', lr=[0.001], clipvalue=1,
    sgd_momentum=0.0, connection_before_conv=False):
    
    ####################
    # Initialize model #
    ####################
    
    x = inputs = Layers.Input(img_shape)
    
    skips = []
    levels_padding = []
    
    activation = Layers.LeakyReLU(alpha=0.2)
    
    for i in range(num_levels):
        
        bottom_pad = 0 if x.shape[1] % 2 == 0 else 1
        right_pad = 0 if x.shape[2] % 2 == 0 else 1
        
        if bottom_pad == 1 or right_pad == 1:
            zero_padding = ((0, bottom_pad), (0, right_pad))
            x = Layers.ZeroPadding2D(zero_padding)(x)
            levels_padding.append(zero_padding)
        else:
            levels_padding.append(None)
            
        x, skip = downsampling(
            x=x, level=i,
            filters=filter_size_start * (2 ** i),
            kernel_size=kernel_size,
            num_convs=num_layers,
            activation=activation,
            batch_norm=batch_norm,
            regularizer=regularizer,
            regularizer_param=regularizer_param,
            connection_before_conv=connection_before_conv)
            
        skips.append(skip)
        
    x = bottleneck(
        x=x,
        filters=filter_size_start * (2 ** num_levels),
        kernel_size=kernel_size,
        num_convs=num_bottleneck,
        activation=activation,
        regularizer=regularizer,
        regularizer_param=regularizer_param,
        batch_norm=batch_norm,
        use_dilation=use_dilation,
        conv2d_activation=conv2d_activation,
        last_activation=last_activation,
        add_skip_layers=add_skip_layers)
        
    for j in range(num_levels):
            
        x = upsampling(
            x=x, level=j, skip=skips[num_levels - j - 1],
            filters=filter_size_start * (2 ** (num_levels - j - 1)),
            kernel_size=kernel_size,
            num_convs=num_layers,
            activation=activation,
            batch_norm=batch_norm,
            regularizer=regularizer,
            regularizer_param=regularizer_param)
        
        if levels_padding[num_levels - j - 1] is not None:
            x = Layers.Cropping2D(cropping=levels_padding[num_levels - j - 1])(x)
        
    outputs = Layers.Conv2D(
        filters=num_classes, kernel_size=1, strides=1,
        padding='same', activation='linear', name = 'linear')(x)
        
    model = Models.Model(inputs=inputs, outputs=outputs)
    
    #########################
    # Process learning rate #
    #########################
    
    if len(lr) == 4:
        shrink_factor = lr[3]

        lr = tfa.optimizers.CyclicalLearningRate(
            initial_learning_rate=lr[0],
            maximal_learning_rate=lr[1],
            step_size=lr[2],
            scale_fn=lambda x: 1/((1. * shrink_factor)**(x-1)))
        
    elif len(lr) == 1:
        lr = lr[0]
    else:
        raise Exception('Unexpected length of lr [{}]'.format(len(lr)))
    
    #####################
    # Process optimizer #
    #####################
    
    if optimizer == 'adam':
        optimizer = tf.optimizers.Adam(learning_rate=lr, amsgrad=False, clipvalue=clipvalue)
    elif optimizer == 'amsgrad':
        optimizer = tf.optimizers.Adam(learning_rate=lr, amsgrad=True, clipvalue=clipvalue)
    elif optimizer == 'adagrad':
        optimizer = tf.optimizers.Adagrad(learning_rate=lr, clipvalue=clipvalue)
    elif optimizer == 'sgd':
        optimizer = tf.optimizers.SGD(learning_rate=lr, momentum=sgd_momentum, clipvalue=clipvalue)
    else:
        raise Exception('Unknown optimizer: {}'.format(optimizer))
    
    model.compile(optimizer=optimizer, loss=cost_func)
    
    return model




def model_unet_nocompile(
    img_shape, num_classes, num_levels, 
    num_layers, num_bottleneck, 
    cost_func, filter_size_start,
    batch_norm=False, kernel_size=3,
    use_dilation=True, conv2d_activation=True,
    last_activation=False, add_skip_layers=True,
    regularizer=None, regularizer_param=0.001,
    optimizer='adam', lr=[0.001], clipvalue=1,
    sgd_momentum=0.0, connection_before_conv=False):
    
    ####################
    # Initialize model #
    ####################
    
    x = inputs = Layers.Input(img_shape)
    
    skips = []
    levels_padding = []
    
    activation = Layers.LeakyReLU(alpha=0.2)
    
    for i in range(num_levels):
        
        bottom_pad = 0 if x.shape[1] % 2 == 0 else 1
        right_pad = 0 if x.shape[2] % 2 == 0 else 1
        
        if bottom_pad == 1 or right_pad == 1:
            zero_padding = ((0, bottom_pad), (0, right_pad))
            x = Layers.ZeroPadding2D(zero_padding)(x)
            levels_padding.append(zero_padding)
        else:
            levels_padding.append(None)
            
        x, skip = downsampling(
            x=x, level=i,
            filters=filter_size_start * (2 ** i),
            kernel_size=kernel_size,
            num_convs=num_layers,
            activation=activation,
            batch_norm=batch_norm,
            regularizer=regularizer,
            regularizer_param=regularizer_param,
            connection_before_conv=connection_before_conv)
            
        skips.append(skip)
        
    x = bottleneck(
        x=x,
        filters=filter_size_start * (2 ** num_levels),
        kernel_size=kernel_size,
        num_convs=num_bottleneck,
        activation=activation,
        regularizer=regularizer,
        regularizer_param=regularizer_param,
        batch_norm=batch_norm,
        use_dilation=use_dilation,
        conv2d_activation=conv2d_activation,
        last_activation=last_activation,
        add_skip_layers=add_skip_layers)
        
    for j in range(num_levels):
            
        x = upsampling(
            x=x, level=j, skip=skips[num_levels - j - 1],
            filters=filter_size_start * (2 ** (num_levels - j - 1)),
            kernel_size=kernel_size,
            num_convs=num_layers,
            activation=activation,
            batch_norm=batch_norm,
            regularizer=regularizer,
            regularizer_param=regularizer_param)
        
        if levels_padding[num_levels - j - 1] is not None:
            x = Layers.Cropping2D(cropping=levels_padding[num_levels - j - 1])(x)
            
    outputs = x
        
#     outputs = Layers.Conv2D(
#         filters=num_classes, kernel_size=1, strides=1,
#         padding='same', activation='linear', name = 'linear')(x)
        
#     model = Models.Model(inputs=inputs, outputs=outputs)
#     #########################
#     # Process learning rate #
#     #########################
    
#     if len(lr) == 4:
#         shrink_factor = lr[3]

#         lr = tfa.optimizers.CyclicalLearningRate(
#             initial_learning_rate=lr[0],
#             maximal_learning_rate=lr[1],
#             step_size=lr[2],
#             scale_fn=lambda x: 1/((1. * shrink_factor)**(x-1)))
        
#     elif len(lr) == 1:
#         lr = lr[0]
#     else:
#         raise Exception('Unexpected length of lr [{}]'.format(len(lr)))
    
#     #####################
#     # Process optimizer #
#     #####################
    
#     if optimizer == 'adam':
#         optimizer = tf.optimizers.Adam(learning_rate=lr, amsgrad=False, clipvalue=clipvalue)
#     elif optimizer == 'amsgrad':
#         optimizer = tf.optimizers.Adam(learning_rate=lr, amsgrad=True, clipvalue=clipvalue)
#     elif optimizer == 'adagrad':
#         optimizer = tf.optimizers.Adagrad(learning_rate=lr, clipvalue=clipvalue)
#     elif optimizer == 'sgd':
#         optimizer = tf.optimizers.SGD(learning_rate=lr, momentum=sgd_momentum, clipvalue=clipvalue)
#     else:
#         raise Exception('Unknown optimizer: {}'.format(optimizer))
    
#     model.compile(optimizer=optimizer, loss=cost_func)
    
    return inputs,outputs,skips


def model_unet_from_file(
    model_path,
    return_cost_func=False,
    model_input_shape=None,
    train_batches=None,
    load_weights=True,
    verbose=True):

    model_dir = os.path.dirname(model_path)

    # Read configuration files
    config_path = os.path.join(model_dir, 'config.yaml')
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Read training summary file if necessary
    if model_input_shape is None or train_batches is None:
        
        summary_path = os.path.join(model_dir, 'model_summary.pkl')

        with open(summary_path, 'rb') as f:
            summary = pickle.load(f)

        assert len(np.intersect1d(summary.keys(), config.keys())) == 0, 'Duplicated names found!'
        
        if model_input_shape is None:
            model_input_shape = (summary['height'], summary['width'], summary['channel'])
            
        if train_batches is None:
            train_batches = summary['n_batches']

    # Get the arguments from the initializer
    static_model_module = SourceFileLoader(
        'model', os.path.join(model_dir, 'model.py')).load_module()

    model_initializer = static_model_module.model_unet
    
    sig = inspect.signature(model_initializer)
    required_keys = list(sig.parameters.keys())

    # Deal with name differences
    config_mapping = {
        'num_classes': 'num_output_channels',
        'num_levels': 'num_unet_levels',
        'num_layers': 'num_convs_per_level',
        'num_bottleneck': 'num_convs_in_bottleneck',
        'mask_output': 'mask_model_output',
    }

    # Manual configuration
    args = {
        'img_shape': model_input_shape,
        'cost_func': utils_cost.get_cost_function(config['cost_func'])
    }

    for name in args:
        required_keys.remove(name)

    # Auto configuration
    for name in required_keys:
        if name in config:
            args[name] = config[name]
        elif name in config_mapping:
            args[name] = config[config_mapping[name]]
        else:
            raise Exception('{} not found!'.format(name))
            
    if args['regularizer'] is not None and verbose:
        print('The loaded model uses regularization ({})!'.format(args['regularizer']))
        
    if config['use_cyclic_lr']:
        args['lr'][2] *= train_batches
        
    model = model_initializer(**args)
    
    if load_weights:
        
        if verbose:
            print('Load model weights ...')
            
        model.load_weights(model_path)
    else:
        if verbose:
            print('No model weights are loaded!')
    
    if return_cost_func:
        return model, args['cost_func']
    else:
        return model

    
def identify_best_model(model_dir, verbose):
    
    if verbose:
            print('Identifying best model from folder {} ...'.format(model_dir))
        
    # Get all available checkpoint files
    model_file = np.array([i.rstrip('.index')
                           for i in glob.glob(os.path.join(model_dir, '*.index'))])
    
    model_file.sort()

    # Extract validation loss
    pattern = '.*?/\d+-(.*?)\.ckpt'
    regex = re.compile(pattern)

    val_loss = np.array([float(regex.match(i).group(1)) for i in model_file])

    # Remove nan if any
    remove_nan = ~np.isnan(val_loss)
    val_loss = val_loss[remove_nan]
    model_file = model_file[remove_nan]

    assert len(model_file) > 0, 'Not valid model file found after removing NAN'

    # Identify the best loss and the position
    best_val_loss = np.min(val_loss)
    
    # If there are multiple, use the model from the last iteration
    best_index = np.where(val_loss == best_val_loss)[0][-1]

    # Identify the best model file
    model_file = model_file[best_index]
    
    return model_file


def reload_model(model_file, loader=None, load_weights=True, verbose=True, return_model_file=False):

    ###################################################
    # Identify the best model if input is a directory #
    ###################################################
    
    # Strip extension if needed
    if os.path.isfile(model_file):
        new_file, ext = os.path.splitext(model_file)
        model_file = new_file if ext == '.index' else model_file
    
    elif os.path.isdir(model_file):
        model_file = identify_best_model(model_dir=model_file, verbose=verbose)
    
    else:
        raise Exception('Non-existing directory or file: {}'.format(model_file))
        
    if verbose:
        print('Reading from {} ...'.format(model_file))
    
    ################
    # Reload model #
    ################

    model, cost_func = model_unet_from_file(
        model_path=model_file,
        return_cost_func=True,
        model_input_shape=None if loader is None else (loader.height, loader.width, len(loader.predictors)),
        train_batches=None if loader is None else loader.n_batches,
        load_weights=load_weights,
        verbose=verbose)

    if verbose:
        print('Done!')
        
    if return_model_file:
        return model, cost_func, model_file
    else:
        return model, cost_func


def unet_predict(model, wrf_scaled, config, subset_with_loader=None, verbose=True, return_everything=False):
    
    # Forward prediction
    if verbose:
        print('Forward prediction with model ...')
        
    pred = model.predict(wrf_scaled)
    returns = {}
    
    assert 'CSGD_MIN_SIGMA_RATIO' in os.environ
    min_sigma_ratio = float(os.environ['CSGD_MIN_SIGMA_RATIO'])
    
    if verbose:
        print('Calculating mu and sigma ...')
    
    if config['cost_func'] in ['masked_crps', 'crps']:
        returns['mu'] = pred[:, :, :, [0]]
        returns['sigma'] = np.abs(pred[:, :, :, [1]]) + 1e-5
        
    elif config['cost_func'] in ['masked_csgd', 'csgd']:
        returns['shift'] = -(np.square(pred[:, :, :, [0]]) + 1e-2)
        returns['unshifted_mu'] = np.square(pred[:, :, :, [1]]) - returns['shift'] + 1e-2
        returns['sigma'] = np.square(pred[:, :, :, [2]]) + returns['unshifted_mu'] * min_sigma_ratio
        returns['mu'] = returns['unshifted_mu'] + returns['shift']

    elif config['cost_func'] in ['masked_ign', 'ign']:
        returns['pop'] = np.square(pred[:, :, :, [0]]) / (1 + np.square(pred[:, :, :, [0]]))
        returns['mu'] = np.square(pred[:, :, :, [1]])
        returns['sigma'] = np.square(pred[:, :, :, [2]])

    else:
        raise Exception('Unknown cost function: {}'.format(config['cost_func']))
    
    if subset_with_loader is not None:
        
        if verbose:
            print('Applying masks with loader ...')
        
        for k in returns.keys():
            tmp = subset_with_loader.apply_mask(returns[k])
            returns[k] = subset_with_loader.reconstruct_domain(tmp)
        
    if return_everything:
        if verbose:
            print('Return a dictionary with {}'.format(returns.keys()))
        return returns

    else:
        if verbose:
            print('Return mu {}, and sigma {}'.format(returns['mu'].shape, returns['sigma'].shape))
        return returns['mu'], returns['sigma']

def bcorre(preddy,input_nc_file):
    
    forho = input_nc_file.split('_')[-1].split('.nc')[0]
    
    if forho=='F000':
        pm = preddy[:,:,0]
        be_sp = np.shape(pm)
        pm_fl = np.ndarray.flatten(pm)
        pm_fl=pm_fl*1.06 + (14.4395)
        outer = np.reshape(pm_fl,[be_sp[0],be_sp[1]])
        preddy[:,:,:,0] = outer
        
    if forho=='F003':
        pm = preddy[:,:,0]
        be_sp = np.shape(pm)
        pm_fl = np.ndarray.flatten(pm)
        pm_fl=pm_fl*1.055 + (11.4395) #learned from validation data
        outer = np.reshape(pm_fl,[be_sp[0],be_sp[1]])
        preddy[:,:,0] = outer
        
    if forho=='F006':
        pm = preddy[:,:,0]
        be_sp = np.shape(pm)
        pm_fl = np.ndarray.flatten(pm)
        pm_fl=pm_fl*1.02 + (3.375) #learned from validation data
        outer = np.reshape(pm_fl,[be_sp[0],be_sp[1]])
        preddy[:,:,0] = outer
        
    return preddy
    

if __name__ == '__main__':
    
    img_height = 65
    img_width = 33
    num_channels = 4
    img_shape = (img_height, img_width, num_channels)
    
    num_classes = 2

    model = model_unet(
        img_shape=img_shape, num_classes=num_classes, num_levels=1,
        num_layers=2, num_bottleneck=3, filter_size_start=16,
        cost_func=utils_cost.crps_cost_function,
        kernel_size=3, bottleneck_dilation=True,
        bottleneck_sum_activation=False)
    
    model.summary()