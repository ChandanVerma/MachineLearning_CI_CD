import numpy as np 
import pandas as pd 

from sklearn.base import BaseEstimator, TransformerMixin

from regression_model.processing import errors

## Categorical imputation
class CategoricalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables = None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y = None):
        # We need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()

        for feature in self.variables:
            X[feature] = X[feature].fillna('Missing')

        return X

## Numerical Imputation
class NumericalImputer(BaseEstimator. TransformerMixin):

    def __init__(self, variables = None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y = None):

        self.imputer_dict = {}
        for feature in self.variables:
            self.imputer_dict[feature] = X[feature].mode()[0]
        return self

    def transform(self, X):

        X = X.copy()
        for feature in self.variables:
            X[feature].fillna(self.imputer_dict[feature], inplace = True)

        return X


class TemporalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables = None, reference_variables = None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

        self.reference_variables = reference_variables

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        X = X.copy()

        for feature in self.variables:
            X[feature] = X[self.reference_variables] - X[feature]

        return X


class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, variables = None, tol = 0.05):

        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
        self.tol = tol

    def fit(self, X, y = None):
        self.encoder_dict_ = {}

        for var in self.variables:
            tmp = pd.Series(X[var].value_counts(normalize = True))
            self.encoder_dict_[var] = list(tmp[tmp >= self.tol].index)

        return self

    def transform(self, X):

        X = X.copy()
        for feature in self.variables:
            X[feature] = np.where(X[feature].isin(self.encoder_dict_[feature]), X[feature], 'Rare')

        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, variables = None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y = None):

        temp = pd.concat(X, y, axis = 1)
        temp.columns = [X.columns] + ['target']

        self.encoder_dict_ = {}

        for var in self.variables:
            t = temp.groupby([var])['target'].mean().sort_values(ascending = False).index
            self.encoder_dict_[var] = {k:i for i, k in enumerate(t, 0)}

        return self

    def transform(self, X):

        X = X.copy()

        for feature in self.variables:
            X[feature] = X[feature].map(self.encoder_dict_[feature])

        return X


class LogTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, variables):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y = None):
        return self

    def transform(self, X):

        X = X.copy()

        for feature in self.variables:
            X[feature] = np.log(X[feature])

        return X

class DropUnecessaryFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, variables = None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        X = X.copy()
        X = X.drop(self.variables, axis = 1)

        return X

