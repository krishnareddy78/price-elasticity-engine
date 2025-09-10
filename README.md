# Advanced Price Elasticity Engine & Optimizer

This project is a production-ready, end-to-end system for solving one of the most critical business problems: pricing strategy. It moves beyond simple elasticity analysis to perform **profit optimization**, a key differentiator from standard portfolio projects.

The engine simulates realistic sales data with confounding variables like seasonality and promotions, then trains a suite of models—from interpretable Log-Log regression to flexible GAMs and powerful XGBoost models—to capture the complex price-demand relationship. The core of the project is a REST API built with FastAPI that takes a product's marginal cost and returns the optimal, profit-maximizing price. The entire application is containerized with Docker and includes a full suite of unit tests with CI/CD via GitHub Actions, demonstrating a commitment to building robust and deployable data science products.

## Project Workflow

1.  **Realistic Data Simulation**: A simulator generates sales data with multiple real-world factors: price fluctuations, seasonality, holidays, and promotional events.
2.  **Multi-Model Training & Evaluation**: The system trains and evaluates a suite of models to capture the price-demand relationship, each with different trade-offs:
    * **Log-Log Linear Model**: A classic, interpretable econometric approach.
    * **GAM (Generalized Additive Model)**: A flexible model to capture non-linear relationships.
    * **XGBoost**: A powerful gradient boosting model to capture complex interactions.
3.  **Profit Optimization API**: A FastAPI server exposes an endpoint that takes a model choice and a product's marginal cost, then returns the profit-maximizing price. This directly answers the core business question: "What price should I set?"
4.  **Containerization**: The entire application is containerized with Docker, ensuring reproducible and seamless deployment.
5.  **Unit Testing & CI**: The project includes a suite of `pytest` unit tests and a GitHub Actions workflow for continuous integration, guaranteeing code reliability.

## Economic & Statistical Foundation

While a simple log-log model provides a direct elasticity estimate ($\ln(Q) \sim \beta_1 \ln(P)$), its accuracy can be compromised by omitted-variable bias. This engine controls for such factors by incorporating them into the models:

<div align="center">

$\ln(Q_d) = \beta_0 + \beta_1 \ln(P) + \beta_2 \cdot \text{is\_promo} + f(\text{day\_of\_year}) + \epsilon$

</div>

Here, $f(\text{day\_of\_year})$ represents a smooth function for seasonality, handled effectively by a GAM.

The ultimate business goal is often profit, not revenue. Profit ($\pi$) is defined as:

<div align="center">

$\pi(P) = (P - C) \cdot Q(P)$

</div>

Where $P$ is price, $C$ is marginal cost, and $Q(P)$ is the quantity predicted by our model at that price. Our API finds the price $P$ that maximizes this function.

## How to Use

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/krishnareddy78/price-elasticity-engine.git
    cd price-elasticity-engine
    ```

2.  **Build and run with Docker (Recommended):**
    This is the simplest way to run the entire application.
    ```bash
    docker-compose up --build
    ```
    The API will be available with interactive documentation at `http://localhost:8000/docs`.

3.  **Run Tests (Locally):**
    ```bash
    pip install -r requirements.txt
    pytest
    ```

## Project Structure
