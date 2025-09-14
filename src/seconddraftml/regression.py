import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import validate_data, check_array, check_is_fitted
from sklearn.ensemble import RandomForestRegressor

from typing import List, Optional, Dict
import random
import copy


class NRandomForestRegressor(RegressorMixin, BaseEstimator):
    """
    Meta-regressor that fits a couple Random Forest models.
    It is designed to train multiple Random Forest models with different hyperparameters for prototype purposes.
    Hyperparameters can be specified by the user or randomly generated.

    Parameters
    ----------
    n_models : int, default=5
        Number of Random Forest models to fit.
    n_estimators : list of int, optional
        List of number of trees for each Random Forest model. If None, random values between 50 and 200 will be generated.
    max_depths : list of int, optional
        List of maximum depths for each Random Forest model. If None, random values between 2 and 64 will be generated.
    random_state : int, optional
        Random seed for reproducibility. If None, the random state is not set.
    other_base_models_kwargs : dict, optional
        Additional keyword arguments to pass to the RandomForestRegressor constructor.

    Attributes
    ----------
    models_ : list of RandomForestRegressor
        List of fitted Random Forest models.
    model_names_ : list of str
        List of names for each Random Forest model based on the number of estimators and max depths
    """

    def __init__(
        self,
        n_models: int = 5,
        n_estimators: Optional[List[int]] = None,
        max_depths: Optional[List[int]] = None,
        other_base_models_kwargs: Optional[List[Dict]] = None,
        random_state: Optional[int] = None,
    ):
        self.n_models = n_models
        self.n_estimators = n_estimators
        self.max_depths = max_depths
        self.other_base_models_kwargs = other_base_models_kwargs
        self.random_state = random_state

    def fit(self, X, y, **kwargs):
        self._validate_params()
        X, y = validate_data(X, y)

        models = []
        for i in range(self.n_models):
            model = RandomForestRegressor(
                n_estimators=self.n_estimators[i],
                max_depth=self.max_depths[i],
                **self.other_base_models_kwargs.get(i),
                random_state=self.random_state,
            )
            model.fit(X, y)
            models.append(model)

        self.models_ = models
        self.model_names_ = [
            f"rf_{n}_{d}" for n, d in zip(self.n_estimators, self.max_depths)
        ]

        return self

    def predict(self, X, **kwargs) -> pd.DataFrame:
        check_is_fitted(self)
        X = check_array(X)
        predictions = np.column_stack([model.predict(X) for model in self.models_])
        return pd.DataFrame(predictions, columns=self.model_names)

    def _validate_params(self):
        if self.n_models <= 0:
            raise ValueError("n_models must be a positive integer")

        if self.n_estimators is None:
            self.n_estimators = [random.randint(50, 200) for _ in range(self.n_models)]
        else:
            if len(self.n_estimators) != self.n_models:
                raise ValueError("n_estimators must have the same length as n_models")

        if self.max_depths is None:
            self.max_depths = [random.randint(2, 64) for _ in range(self.n_models)]
        else:
            if len(self.max_depths) != self.n_models:
                raise ValueError("max_depths must have the same length as n_models")

        if self.other_base_models_kwargs is None:
            self.other_base_models_kwargs = [{} for _ in range(self.n_models)]
        else:
            if len(self.other_base_models_kwargs) != self.n_models:
                raise ValueError(
                    "other_base_models_kwargs must have the same length as n_models"
                )
            for kwargs in self.other_base_models_kwargs:
                if not isinstance(kwargs, dict):
                    raise ValueError(
                        "other_base_models_kwargs must be a list of dictionaries"
                    )

    def __sklearn_clone__(self):
        return NRandomForestRegressor(
            n_models=self.n_models,
            n_estimators=self.n_estimators,
            max_depths=self.max_depths,
            # correct copying of mutable parameters for sklearn compatibility
            other_base_models_kwargs=copy.deepcopy(self.other_base_models_kwargs),
            random_state=self.random_state,
        )

    def __sklearn_is_fitted__(self):
        # check if all base models are fitted
        return all(check_is_fitted(model) for model in self.models_)
