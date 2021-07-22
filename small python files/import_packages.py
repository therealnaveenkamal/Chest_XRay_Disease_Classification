import os
import cv2
import random
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from keras import backend as K
from matplotlib import rcParams
from keras.preprocessing import image
from tensorflow.keras.models import Model
from sklearn.preprocessing import OneHotEncoder
from keras.applications.densenet import DenseNet121
from sklearn.metrics import roc_auc_score, roc_curve
from tensorflow.keras.layers import ReLU, concatenate
from tensorflow.compat.v1.logging import INFO, set_verbosity
from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dense
from tensorflow.keras.layers import AvgPool2D, GlobalAveragePooling2D, MaxPool2D

disable_eager_execution()
{"mode":"full","isActive":false}