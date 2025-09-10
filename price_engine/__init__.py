"""
Price Engine Package

A collection of modules for simulating sales data, modeling price elasticity,
and optimizing for profit.
"""
from .data_simulator import simulate_sales_data
from .models import LogLogElasticityModel, GamelasticityModel, XGBoostElasticityModel
from .optimizer import ProfitOptimizer

__version__ = "2.0.0"
