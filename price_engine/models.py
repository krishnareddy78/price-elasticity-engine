import pandas as pd
import numpy as np
import statsmodels.api as sm
from xgboost import XGBRegressor
from pygam import LinearGAM, s
from abc import ABC, abstractmethod

class BaseElasticityModel(ABC):
    """Abstract base class for elasticity models."""
    def __init__(self):
        self.model = None
        self.features = ['price']
        self.is_log_transformed = True # Assume log-log by default

    @abstractmethod
    def fit(self, data: pd.DataFrame):
        """Trains the model on the provided data."""
        pass

    def _prepare_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Prepares the data by log-transforming and selecting features."""
        df = data.copy()
        if self.is_log_transformed:
            df['target'] = np.log(df['quantity_sold'])
            df['price'] = np.log(df['price'])
        else:
            df['target'] = df['quantity_sold']
        
        X = df[self.features]
        y = df['target']
        return X, y

    def predict_quantity(self, data: pd.DataFrame) -> np.ndarray:
        """Predicts quantity sold for a given set of input data."""
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        
        df = data.copy()
        if self.is_log_transformed:
            df['price'] = np.log(df['price'])
        
        X = df[self.features]
        predictions = self.model.predict(X)
        
        if self.is_log_transformed:
            return np.exp(predictions)
        return predictions

class LogLogElasticityModel(BaseElasticityModel):
    """
    Estimates price elasticity using a log-log linear regression with control variables.
    """
    def __init__(self):
        super().__init__()
        self.features = ['price', 'is_promo']
        self.elasticity = None

    def fit(self, data: pd.DataFrame):
        X, y = self._prepare_data(data)
        X = sm.add_constant(X)
        
        ols_model = sm.OLS(y, X)
        self.model = ols_model.fit()
        self.elasticity = self.model.params['price']

class GamelasticityModel(BaseElasticityModel):
    """
    Estimates price elasticity using a Generalized Additive Model (GAM)
    to capture non-linear seasonality.
    """
    def __init__(self):
        super().__init__()
        self.features = ['price', 'is_promo', 'day_of_year']
        self.elasticity = None

    def fit(self, data: pd.DataFrame):
        X, y = self._prepare_data(data)
        # s(2) for day_of_year indicates a smooth term
        self.model = LinearGAM(s(0) + s(1) + s(2)).fit(X, y)

        # Partial dependency to estimate elasticity
        X_copy = X.copy()
        X_copy['price'] += 0.01 # 1% increase
        
        pred_base = self.model.predict(X)
        pred_new = self.model.predict(X_copy)

        percent_change_q = (np.exp(pred_new) - np.exp(pred_base)) / np.exp(pred_base)
        self.elasticity = np.mean(percent_change_q) / 0.01

class XGBoostElasticityModel(BaseElasticityModel):
    """
    Estimates price elasticity using an XGBoost model.
    """
    def __init__(self):
        super().__init__()
        self.features = ['price', 'is_promo', 'day_of_year']

    def fit(self, data: pd.DataFrame):
        X, y = self._prepare_data(data)
        self.model = XGBRegressor(objective='reg:squarederror', n_estimators=200, random_state=42)
        self.model.fit(X, y)

        # Partial dependency to estimate elasticity
        X_copy = X.copy()
        X_copy['price'] += 0.01 # 1% increase
        
        pred_base = self.model.predict(X)
        pred_new = self.model.predict(X_copy)
        
        percent_change_q = (np.exp(pred_new) - np.exp(pred_base)) / np.exp(pred_base)
        self.elasticity = np.mean(percent_change_q) / 0.01
