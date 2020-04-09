import numpy as np 
import pandas as pd 

from sklearn.base import BaseEstimator, TransformerMixin
from regression_model.processing.errors import InvalidModelInputError 

class CategoricalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables = None) -> None:
        if not isinstance(variables, list):
            self.variables = variables
        else:
            self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "CategoricalImputer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        for feature in self.variables:
            X[feature] = X[feature].fillna('Missing')

        return X


class NumericalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables = None) -> None:
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X: pd.DataFrame, y:pd.Series = None) -> None:
        self.imputer_dict_ = {}
        for var in self.variables:
            self.imputer_dict_[var] = X[var].mode()[0]

        return self

    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for feature in self.variables:
            X[feature].fillna(self.imputer_dict_[feature], inplace = True)

        return X 


class TemporalVariableEstimator(BaseEstimator, TransformerMixin):

    def __init__(self, variables = None, reference_variable = None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

        self.reference_variable = reference_variable

    def fit(self, X: pd.Dataframe, y: pd.Series = None):
        return self

    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        for feature in self.variables:
            X[feature] = X[self.reference_variable] = X[feature]

        return X


class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, variables = None, tol = 0.05):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

        self.tol = tol

    def fit(self, X:pd.DataFrame, y:pd.Series = None):
        self.encoder_dict_ = {}
        for var in self.variables:
            t = X[var].value_counts(normalize = True)
            self.encoder_dict_[var] = list(t[t >= self.tol].index)

        return self

    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        for feature in self.variables:
            X[feature] = np.where(X[feature].isin(self.encoder_dict_[feature]), X[feature], "Rare")

        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, variables = None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    
    def fit(self, X:pd.DataFrame, y:pd.Series = None):
        temp = pd.concat(X, y, axis = 1)
        temp.columns = X.columns + ['target']

        self.encoder_dict_ = {}

        for var in self.variables:
            t = temp.groupby(var)['target'].mean().sort_values(ascending = False).index
            self.encoder_dict_[var] = {k:i for i, k in enumerate(t, 0)}

        return self

    def transform(self, X:pd.DataFrame)-> pd.DataFrame:
        X = X.copy()

        for feature in self.variables:
            X[feature] = X[feature].map(self.encoder_dict_[feature])

        if X[self.variables].isnull().any().any():
            null_counts = X[self.variables].isnull().any()
            vars_ = {key: value for (key, value) in null_counts.items() if value is True}
            raise InvalidModelInputError(f"Categorical Variables has introduced NaN when transforming cateforical variables: {vars_.keys()}")

        return X


class DropUnecessaryFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, variables_to_drop = None):
        
        self.variables_to_drop = variables_to_drop

    def fit(self, X: pd.DataFrame, y:pd.Series = None):
        return self

    def transform(self, X:pd.DataFrame):
        
        X = X.copy()
        X = X.drop(self.variables_to_drop, axis = 1)

        return X