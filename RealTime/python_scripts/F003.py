#################################################################################
### Import Packages ### run in tfp environment: 
####################################################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np

import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import tensorflow_probability as tfp
from tensorflow.python.keras.optimizer_v2.adam import Adam
tfd = tfp.distributions
import tensorflow.keras.backend as K
from tensorflow import math as tfm

import os
import utilsProb
import utilsProbSS
import glob
import sys
from scipy.stats import rankdata
import pandas as pd
import importlib
import copy
from netCDF4 import Dataset, num2date
from scipy.interpolate import interpn
from matplotlib.colors import Normalize 
from matplotlib import cm
import matplotlib as mpl
import seaborn as sns
sns.set_style('whitegrid', {'font.family':'serif', 'font.serif':'Times New Roman'})
import properscoring as ps
from math import erf
import xarray as xr
import matplotlib

import Unet_b
import utils_CNN
import coms
from tensorflow.python.client import device_lib
####################################################################################
tf.random.set_seed(33) #patrick roy's number.
####################################################################################

stepnum=2
years_back = 2
input_nc_file='/glade/work/wchapman/ARcnn_LRP_DEMO/RealTime/regrid_folder/IVT_ERAGrid_F003.nc'
norm_file ='/glade/work/wchapman/ARcnn_LRP_DEMO/RealTime/python_scripts/models/F003/norm_dict.npy'
outputfile ='/glade/work/wchapman/ARcnn_LRP_DEMO/RealTime/output_forecast/pp_F003.nc'
Wsave_name='/glade/work/wchapman/ARcnn_LRP_DEMO/RealTime/python_scripts/models/F003/cpf_CRPS_val_2016_test_2017.ckpt'


dirA = ['F'+f'{x:03}' for x in np.arange(0,168,3)]
print('#############################################')
print('post processing forecast:', dirA[stepnum-1])
print('#############################################')
from matplotlib.colors import ListedColormap

import scipy.stats
import model_u
from importlib import reload 
import tensorflow.keras.models as Models
import tensorflow.keras.layers as Layers
import json
#################################################################################
### Import Packages ### run in tfp environment: 
####################################################################################


####################################################################################
#GPU cuda handling: 
####################################################################################
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

print(get_available_gpus())

os.environ["CUDA_VISIBLE_DEVICES"]="0"

if tf.test.gpu_device_name() != '/device:GPU:0':
    print('#################################################')
    print('#################################################')
    print('WARNING: GPU device not found.')
    print('#################################################')
    print('#################################################')
else:
    print('#################################################')
    print('#################################################')
    print('SUCCESS: Found GPU: {}'.format(tf.test.gpu_device_name()))
    print('#################################################')
    print('#################################################')
####################################################################################
print('We are here:',os.getcwd())
os.chdir('/glade/work/wchapman/AnEn/CNN/Coastal_Points_LogNormal/')
####################################################################################

#Build Model
img_shape = (141, 141, 6)
nummy_classes = 2
print(stepnum)

if stepnum > 2:
    ins,outs,skips = model_u.model_unet_nocompile(img_shape=img_shape, cost_func=coms.crps_cost_function,num_classes=nummy_classes,num_levels = 2,num_layers =3, num_bottleneck = 3, filter_size_start =64, batch_norm=None, kernel_size = 2,
                                   use_dilation=True,conv2d_activation=True)
    outs = Layers.Concatenate()([outs, ins])
    outputs = Layers.Conv2D(filters=2, kernel_size=1, strides=1,padding='same', activation='linear', name = 'linear')(outs)
    model = Models.Model(inputs=ins, outputs=outputs)
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001, amsgrad=False, clipvalue=1), loss=coms.crps_cost_function)
    model.summary()
    
else:
    model = model_u.model_unet(img_shape=img_shape, cost_func=coms.crps_cost_function,num_classes=nummy_classes,num_levels = 0,num_layers =3, num_bottleneck = 3, filter_size_start =256, batch_norm=None, kernel_size = 1,use_dilation=True,conv2d_activation=True)
    model.summary()
    
#Load Model Weights
Wsave_name='/glade/work/wchapman/ARcnn_LRP_DEMO/RealTime/python_scripts/models/F003/cpf_CRPS_val_2016_test_2017.ckpt'
model.load_weights(Wsave_name)
print('...model weights loaded...')

print('...grab normalization dictionary...')
norm_dict = np.load(norm_file, allow_pickle=True).flat[0]
print(norm_dict)

print('...use xarray to open file...')
DS = xr.open_dataset(input_nc_file)
DS = DS.sel(pressure=500).squeeze()
DS=DS[['IVT','p_sfc','u_tr_p','v_tr_p','Z_p','IWV']].squeeze()
print(DS)

print('normalize')
for nn,kee in enumerate(norm_dict):    
    if nn ==0:
        ictor_array = np.expand_dims((np.array(DS[kee].squeeze())-norm_dict[kee][0])/norm_dict[kee][1],axis=2)
    else: 
        ictor_array = np.concatenate([ictor_array,np.expand_dims((np.array(DS[kee].squeeze())-norm_dict[kee][0])/norm_dict[kee][1],axis=2)],axis=2)
        
preddy = np.abs(model.predict(np.expand_dims(ictor_array,axis=0)).squeeze())
#bcorrect: 
preddy=model_u.bcorre(preddy,input_nc_file)
print(preddy.shape,'...prediction...')

muf  = preddy[:,:,0]
stdf = preddy[:,:,1]
lat =np.array(DS.lat)
lon =np.array(DS.lon)

prob_greater_250 = np.zeros_like(muf)
threshy = 250
for lala in range(lat.shape[0]):
    for lolo in range(lon.shape[0]):
        prob_greater_250[lala,lolo]=scipy.stats.norm.sf(threshy,muf[lala,lolo],np.abs(stdf[lala,lolo]))
        
prob_greater_500 = np.zeros_like(muf)
threshy = 500
for lala in range(lat.shape[0]):
    for lolo in range(lon.shape[0]):
        prob_greater_500[lala,lolo]=scipy.stats.norm.sf(threshy,muf[lala,lolo],np.abs(stdf[lala,lolo]))
        
        
prob_greater_750 = np.zeros_like(muf)
threshy = 750
for lala in range(lat.shape[0]):
    for lolo in range(lon.shape[0]):
        prob_greater_750[lala,lolo]=scipy.stats.norm.sf(threshy,muf[lala,lolo],np.abs(stdf[lala,lolo]))
        

#Create Xarray Dataset...and save.
NU_ds2 = xr.Dataset(
    {
        "IVT_mean":(["lat","lon"],(preddy[:,:,0])), 
        "IVT_sd":(["lat","lon"],(preddy[:,:,1])), 
        "WWRF_IVT":(["lat","lon"],(DS['IVT'])), 
        "prob_greater_250":(["lat","lon"],(prob_greater_250)), 
        "prob_greater_500":(["lat","lon"],(prob_greater_500)), 
        "prob_greater_750":(["lat","lon"],(prob_greater_750)) 
    },
    coords={
        "time":DS.time,
        "lat":np.array(DS.lat),
        "lon":np.array(DS.lon),
    },)

print(NU_ds2)
NU_ds2.to_netcdf(outputfile)