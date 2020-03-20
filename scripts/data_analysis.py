import warnings
warnings.filterwarnings('ignore')
import os
import sys
from pathlib import Path

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

import yaml

with open('config.yaml', 'r') as f:
    config = yaml.load(f)

config['PATH']['PROJECT_PATH']