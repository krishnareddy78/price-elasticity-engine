import numpy as np
import pandas as pd

class ProfitOptimizer:
    """
    Calculates the optimal price to maximize profit.
    """
    def __init__(self, model, baseline_data: pd.DataFrame):
        self.model = model
        # Use baseline data to provide average values for non-price features
        self.baseline = baseline_data[model.features].drop('price', axis=1).mean().to_dict()

    def find_optimal_price(self, marginal_cost: float, price_range: tuple) -> dict:
        """
        Finds the profit-maximizing price within a given range.

        Args:
            marginal_cost: The cost to produce one additional unit.
            price_range: A tuple (min_price, max_price) to search within.
        """
        prices = np.linspace(price_range[0], price_range[1], 200)
        
        # Create a dataframe for prediction with the search prices
        search_df = pd.DataFrame({'price': prices})
        for feature, value in self.baseline.items():
            search_df[feature] = value
        
        # Predict quantities
        quantities = self.model.predict_quantity(search_df)
        
        # Calculate profit
        profits = (prices - marginal_cost) * quantities
        
        # Find the maximum
        optimal_index = np.argmax(profits)
        optimal_price = prices[optimal_index]
        max_profit = profits[optimal_index]
        
        return {
            "estimated_elasticity": round(self.model.elasticity, 4),
            "marginal_cost": round(marginal_cost, 2),
            "optimal_price": round(optimal_price, 2),
            "predicted_max_profit": round(max_profit, 2)
        }
