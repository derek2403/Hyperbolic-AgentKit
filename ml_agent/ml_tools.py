from langchain.tools import Tool
from ml_agent.ml_actions import train_ml_model

def create_ml_training_tool() -> Tool:
    """Create a tool for ML model training."""
    return Tool(
        name="train_ml_model",
        description="""Train a machine learning model using a Kaggle dataset.
        Required inputs:
        - dataset_url: Kaggle dataset URL
        - target_column: Name of the column to predict
        - model_type: One of [linear_regression, ridge, lasso, elastic_net, random_forest, gradient_boosting, svr, xgboost, k_neighbors]
        
        Example: train_ml_model(
            dataset_url="https://www.kaggle.com/datasets/example/dataset",
            target_column="price",
            model_type="random_forest"
        )""",
        func=lambda dataset_url, target_column, model_type: train_ml_model(
            dataset_url=dataset_url,
            target_column=target_column,
            model_type=model_type
        )
    )
