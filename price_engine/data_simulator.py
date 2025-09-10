import numpy as np
import pandas as pd

def simulate_sales_data(
    base_demand: int = 1000,
    price_elasticity: float = -2.5,
    base_price: float = 20.0,
    num_days: int = 365 * 2,
    noise_std: float = 0.1,
    seasonality_strength: float = 0.4,
    promo_effect: float = 0.5,
    holiday_effect: float = 0.6
) -> pd.DataFrame:
    """
    Generates realistic daily sales data with multiple confounding factors.
    This provides a challenging dataset for models to learn from.
    """
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=num_days, freq='D'))
    
    # Simulate price variations
    price_fluctuations = np.random.normal(loc=0, scale=0.07, size=num_days)
    prices = base_price * (1 + price_fluctuations)
    prices = np.clip(prices, base_price * 0.75, base_price * 1.25)

    # Calculate baseline demand from price
    log_base_demand = np.log(base_demand)
    log_demand = log_base_demand + price_elasticity * (np.log(prices) - np.log(base_price))
    
    # Add control variables (confounders)
    day_of_year = dates.dayofyear
    seasonal_effect = seasonality_strength * np.sin(2 * np.pi * (day_of_year - 80) / 365.25) # Peak in spring/summer
    
    # Simulate promotions every ~45 days
    is_promo = (day_of_year % 45 == 0).astype(int)
    promo_lift = is_promo * promo_effect
    
    # Simulate major holidays
    is_holiday = (dates.month == 12) & (dates.day > 15) | (dates.month == 11) & (dates.day > 20)
    holiday_lift = is_holiday.astype(int) * holiday_effect

    # Add random noise
    random_noise = np.random.normal(loc=0, scale=noise_std, size=num_days)

    # Combine all effects
    final_log_demand = log_demand + seasonal_effect + promo_lift + holiday_lift + random_noise
    quantity_sold = np.round(np.exp(final_log_demand)).astype(int)
    quantity_sold = np.maximum(quantity_sold, 10)

    df = pd.DataFrame({
        'date': dates,
        'price': np.round(prices, 2),
        'quantity_sold': quantity_sold,
        'is_promo': is_promo,
        'day_of_year': day_of_year
    })
    return df
