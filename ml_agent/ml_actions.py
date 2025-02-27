from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Callable, Any
from hyperbolic_agentkit_core.actions.hyperbolic_action import HyperbolicAction
from ml_agent.ml_trainer import MLTrainer

class MLTrainingInput(BaseModel):
    """Input schema for ML training."""
    dataset_url: str = Field(
        ..., 
        description="Kaggle dataset URL (e.g., 'https://www.kaggle.com/datasets/example/dataset')"
    )
    target_column: str = Field(
        ..., 
        description="Name of the target column to predict"
    )
    model_type: str = Field(
        ..., 
        description="""Type of model to train. Options:
        - linear_regression
        - ridge
        - lasso
        - elastic_net
        - random_forest
        - gradient_boosting
        - svr
        - xgboost
        - k_neighbors"""
    )

def train_ml_model(dataset_url: str, target_column: str, model_type: str) -> str:
    """
    Train an ML model using the specified parameters.
    
    Args:
        dataset_url: Kaggle dataset URL
        target_column: Name of the target column
        model_type: Type of model to train
        
    Returns:
        str: Formatted results of the training process
    """
    trainer = MLTrainer()
    results = trainer.train_model_pipeline(dataset_url, target_column, model_type)
    
    if results["status"] == "error":
        return f"Error training model: {results['error_message']}"
    
    # Format the results
    output = [
        "Model Training Results:",
        "-------------------",
        f"Dataset Info:",
        f"- Rows: {results['dataset_info']['rows']}",
        f"- Columns: {', '.join(results['dataset_info']['columns'])}",
        f"\nModel Info:",
        f"- Type: {results['model_info']['type']}",
        f"\nMetrics:",
        f"- MAE: {results['metrics']['MAE']:.4f}",
        f"- MSE: {results['metrics']['MSE']:.4f}",
        f"- R²: {results['metrics']['R²']:.4f}"
    ]
    
    return "\n".join(output)

class MLTrainingAction(HyperbolicAction):
    """ML model training action."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str = "train_ml_model"
    description: str = """Train a machine learning model using a Kaggle dataset.
    Supports various regression models and automatic hyperparameter tuning.
    Requires a Kaggle API key to be set up."""
    args_schema: type[BaseModel] = MLTrainingInput
    func: Callable[[str, str, str], str] = train_ml_model
