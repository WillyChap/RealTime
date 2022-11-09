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

import yaml

with open('../config_for.yaml') as f:
    yaml_dict = yaml.safe_load(f)
####################################################################################
tf.random.set_seed(33) #patrick roy's number.
####################################################################################

print(yaml_dict['path_root_information']['root_loc'])


