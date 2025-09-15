import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import validate_data, check_array, check_is_fitted
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge

from typing import List, Optional, Dict
import random

class HyperParameterSamplerMixin:
    """
    Mixin class to sample hyperparameters for well-known sklearn models.
    """
    
    def sample_hyperparameters(self, model_type: BaseEstimator, n_samples: int = 5) -> List[Dict]:
        """
        Sample hyperparameters for a given sklearn model.

        Parameters
        ----------
        model_name : str
            Name of the sklearn model.
        n_samples : int, default=1
            Number of samples to generate.

        Returns
        -------
        List[Dict]
            List of dictionaries containing sampled hyperparameters.
        """

        # RFs and ETs, focus on max_depth from 2 to 32 and n_estimators from 50 to 200
        if model_type == RandomForestRegressor or model_type == ExtraTreesRegressor:
            samples = [
                {
                    "n_estimators": random.randint(50, 200),
                    "max_depth": random.randint(2, 32),
                }
                for _ in range(n_samples)
            ]
        # Ridge, focus on alpha from 0.001 to 10 log uniform
        elif model_type == Ridge:
            samples = [
                {
                    "alpha": 10 ** random.uniform(-3, 1),
                }
                for _ in range(n_samples)
            ]
        return samples
        

class FeatureImportanceMixin:
    """
    Mixin class to provide feature importance for regression models.
    """

    def feature_importance(self, model) -> pd.DataFrame:
        """
        Calculate feature importance for the fitted model.

        Parameters
        ----------
        X : pd.DataFrame
            Input features used for training the model.

        Returns
        -------
        pd.DataFrame
            DataFrame containing feature importances.
        """
        check_is_fitted(model)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = self.coef_
        importances_df = pd.DataFrame({
            'feature': self.feature_names_in_,
            'importance': importances
        })
        importances_df = importances_df.sort_values(
            by='importance', ascending=False, key=lambda x: abs(x)
        ).reset_index(drop=True)
        return importances_df


class RegressorList(HyperParameterSamplerMixin, FeatureImportanceMixin, RegressorMixin, BaseEstimator):
    """
    A class to manage a list of regression models with sampled hyperparameters.
    Parameters
    ----------
    base_model_class : BaseEstimator, optional
        The base regression model class to sample hyperparameters from.
    n_models : int, default=5
        The number of models to create with sampled hyperparameters.
    """

    def __init__(self, base_model_class: BaseEstimator, n_models: int = 5):
        print(base_model_class)
        self.base_model_class = base_model_class
        self.n_models = n_models

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'RegressorList':
        X, y = validate_data(self, X, y)
        hyperparams = self.sample_hyperparameters(self.n_models)
        models = []
        for hyperparam in hyperparams:
            model = self.base_model_class(**hyperparam)
            model.fit(X, y)
            models.append(model)
        self.models_ = models
        self.model_names_ = self.get_models_names(hyperparams)
        return self

    def predict(self, X :pd.DataFrame, **kwargs) -> pd.DataFrame:
        check_is_fitted(self)
        X = check_array(X)
        predictions = np.column_stack([model.predict(X) for model in self.models_])
        return pd.DataFrame(predictions, columns=self.model_names_)

    def sample_hyperparameters(self, n_samples: int = 5) -> List[Dict]:
        return super().sample_hyperparameters(self.base_model_class, n_samples)
    
    def get_models_names(self, hyperparams) -> List[str]:
        model_name = self.base_model_class.__name__
        model_names = []
        for hyperparam in hyperparams:
            model_params = "_".join(f"{k}_{str(v)}" for k, v in hyperparam.items())
            model_names.append(f"{model_name}_{model_params}")
        return model_names
    
    def feature_importance(self) -> List[Dict]:
        return [
            {
                "model_name": model_name,
                "importance": super().feature_importance(model)
            }
            for model_name, model in zip(self.model_names_, self.models_)
        ]
    
    def __sklearn_is_fitted__(self) -> bool:
        # check if all base models are fitted
        # all_fitted = all(check_is_fitted(model) for model in self.models_)
        return hasattr(self, 'models_')
    

class RandomForestRegressorList(RegressorList):
    """
    A class to manage a list of RandomForestRegressor models with sampled hyperparameters.
    Inherits from RegressorList.
    """
    
    def __init__(self, n_models: int = 5):
        super().__init__(base_model_class=RandomForestRegressor, n_models=n_models)


class ExtraTreesRegressorList(RegressorList):
    """
    A class to manage a list of ExtraTreesRegressor models with sampled hyperparameters.
    Inherits from RegressorList.
    """
    
    def __init__(self, n_models: int = 5):
        super().__init__(base_model_class=ExtraTreesRegressor, n_models=n_models)


class RidgeRegressorList(RegressorList):
    """
    A class to manage a list of Ridge models with sampled hyperparameters.
    Inherits from RegressorList.
    """
    
    def __init__(self, n_models: int = 5):
        super().__init__(base_model_class=Ridge, n_models=n_models)