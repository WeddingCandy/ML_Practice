# -*- coding: utf-8 -*-
"""
@CREATETIME: 24/07/2018 19:29 
@AUTHOR: Chans
@VERSION: 
"""

import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn import preprocessing
import numpy as np
from xgboost import plot_importance
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
import sklearn.feature_selection as fs

train_file = '/Users/Apple/datadata/football_data/train.csv'
test_file = '/Users/Apple/datadata/football_data/test.csv'

pd_train = pd.read_csv(train_file,encoding='utf-8')