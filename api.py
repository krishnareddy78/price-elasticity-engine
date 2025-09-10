from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from price_engine import (
    simulate_sales_data,
    LogLogElasticityModel,
    GamelasticityModel,
    XGBoostElasticityModel,
    ProfitOptimizer
)

# Use a dictionary to act as a simple, in-memory model registry
MODEL_REGISTRY = {}
DATA = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # This code runs on startup
    print("--- Training models on startup ---")
    global DATA
    DATA = simulate_sales_data()
    
    MODEL_REGISTRY["log_log"] = LogLogElasticityModel()
    MODEL_REGISTRY["gam"] = GamelasticityModel()
    MODEL_REGISTRY["xgboost"] = XGBoostElasticityModel()

    for name, model in MODEL_REGISTRY.items():
        print(f"Training {name} model...")
        model.fit(DATA)
        print(f"{name} model trained. Elasticity: {model.elasticity:.4f}")
    
    print("--- Startup complete ---")
    yield
    # This code runs on shutdown (cleanup)
    MODEL_REGISTRY.clear()

def create_app():
    app = FastAPI(
        title="Advanced Price Engine API",
        description="API for profit optimization using multiple elasticity models.",
        version="2.0.0",
        lifespan=lifespan
    )

    @app.get("/", tags=["General"])
    def read_root():
        return {"message": "Welcome to the Advanced Price Engine API. Go to /docs for documentation."}

    @app.get("/models", tags=["Models"])
    def get_available_models():
        """Returns the names and estimated elasticities of available models."""
        return {name: {"estimated_elasticity": model.elasticity} for name, model in MODEL_REGISTRY.items()}

    @app.post("/optimize-profit", tags=["Optimization"])
    def optimize_profit(model_name: str, marginal_cost: float, min_price: float, max_price: float):
        """
        Calculates the profit-maximizing price.
        """
        if model_name not in MODEL_REGISTRY:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")
        if min_price >= max_price:
            raise HTTPException(status_code=400, detail="min_price must be less than max_price.")

        model = MODEL_REGISTRY[model_name]
        optimizer = ProfitOptimizer(model, baseline_data=DATA)
        result = optimizer.find_optimal_price(marginal_cost, (min_price, max_price))
        return result
        
    return app
