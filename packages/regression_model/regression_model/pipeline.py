from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from regression_model.processing import preprocessors as pp 
from regression_model.processing import features as fe 
from regression_model.config import config

import logging
_logger = logging.getLogger(__name__)


price_pipe = Pipeline(
    [
        (
            "categorical_imputer", pp.CategoricalImputer(variables = config.CATEGORICAL_VARS_WITH_NA),
        ),
        (
            "numerical_imputer", pp.NumericalImputer(variables = config.NUMERICAL_VARS_WITH_NA),
        ),
        (
            "temporal_varibales", pp.TemporalVariableEstimator(variables = config.TEMPORAL_VARS),
        ),
        (
            "rare_label_encoder", pp.RareLabelCategoricalEncoder(tol = 0.01, variables = config.CATEGORICAL_VARS),
        ),
        (
            "categorical_encoder", pp.CategoricalEncoder(variables = config.CATEGORICAL_VARS),
        ),
        (
            "log_transform", fe.LogTransformer(variables = config.NUMERICAL_LOG_VARS),
        ),
        (
            "drop_features", pp.DropUnecessaryFeatures(variables_to_drop = config.DROP_FEATURES),
        ),
        (   "scalar", MinMaxScaler()),
        (   "Linear_model", Lasso(alpha = 0.005, random_state = 0)),
    ]
)