import pathlib
import regression_model
import pandas as pd 

pd.options.display.max_rows = 10
pd.options.display.max_columns = 10

PACKAGE_ROOT = pathlib.Path(regression_model.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
DATASET_DIR  = PACKAGE_ROOT / "datasets"

## data
TRAINING_DATA_FILE = 'train.csv'
TESTING_DATA_FILE = 'test.csv'
TARGET = 'SalePrice'

## variables
FEATURES = ["MSSubClass","MSZoning", "Neighborhood", "OverallQual", "OverallCond", "YearRemodAdd", "RoofStyle", "MasVnrType", "BsmtQual", "BsmtExposure", "HeatingQC", "CentralAir", "1stFlrSF", "GrLivArea",
    "BsmtFullBath", "KitchenQual", "Fireplaces", "FireplaceQu", "GarageType", "GarageFinish", "GarageCars", "PavedDrive", "LotFrontage",
    # this one is only to calculate temporal variable:
    "YrSold",
]

DROP_FEATURES = 'YrSold'

NUMERICAL_VARS_WITH_NA = ['LotFrontage']

CATEGORICAL_VARS_WITH_NA = ["MasVnrType", "BsmtQual", "BsmtExposure", "FireplaceQu", "GarageType", "GarageFinish",]

TEMPORAL_VARS = 'YearRemodAdd'

NUMERICAL_LOG_VARS = ["LotFrontage", "1stFlrSF", "GrLivArea"]

CATEGORICAL_VARS = [ "MSZoning", "Neighborhood", "RoofStyle", "MasVnrType", "BsmtQual", "BsmtExposure", "HeatingQC", "CentralAir", "KitchenQual", "FireplaceQu", "GarageType", "GarageFinish", "PavedDrive",]

NUMERICAL_NA_NOT_ALLOWED = [feature for feature in FEATURES if feature not in CATEGORICAL_VARS + NUMERICAL_VARS_WITH_NA]

CATEGORICAL_NA_NOT_ALLOWED = [feature for feature in CATEGORICAL_VARS if feature not in CATEGORICAL_VARS_WITH_NA]

PIPELINE_NAME = 'lasso_regression'
PIPELINE_SAVE_FILE = f"{PIPELINE_NAME}_output_v"

# used for differential testing
ACCEPTABLE_MODEL_DIFFERENCE = 0.05