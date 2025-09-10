import pytest
import pandas as pd
from price_engine import (
    simulate_sales_data,
    LogLogElasticityModel,
    ProfitOptimizer
)

@pytest.fixture(scope="module")
def sales_data():
    """Generate a single dataset for all tests in this module."""
    return simulate_sales_data(num_days=365)

def test_data_simulator_output(sales_data):
    """Test the output of the data simulator."""
    assert isinstance(sales_data, pd.DataFrame)
    assert sales_data.shape[0] == 365
    assert 'quantity_sold' in sales_data.columns
    assert sales_data['quantity_sold'].min() >= 10

def test_log_log_model_fit(sales_data):
    """Test that the LogLog model can be fitted without errors."""
    model = LogLogElasticityModel()
    model.fit(sales_data)
    assert model.elasticity is not None
    assert isinstance(model.elasticity, float)
    # Elasticity should be negative
    assert model.elasticity < 0

def test_optimizer_with_fitted_model(sales_data):
    """Test the profit optimizer with a trained model."""
    model = LogLogElasticityModel()
    model.fit(sales_data)
    
    optimizer = ProfitOptimizer(model, baseline_data=sales_data)
    result = optimizer.find_optimal_price(marginal_cost=10.0, price_range=(15.0, 25.0))
    
    assert 'optimal_price' in result
    assert 'predicted_max_profit' in result
    assert 15.0 <= result['optimal_price'] <= 25.0
