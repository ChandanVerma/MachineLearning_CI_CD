import numpy as np 
import pandas as pd 

from regression_model.processing.data_management import load_pipeline
from regression_model.config import config
from regression_model.processing.validation import validated_inputs
from regression_model import __version__ as _version 

import logging
import typing as t 

_logger = logging.getLogger(__name__)

pipeline_file_name = f"{config.PIPELINE_SAVE_FILE}{_version}.pkl"