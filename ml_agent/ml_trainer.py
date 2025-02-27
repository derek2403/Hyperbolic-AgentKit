import os
import pandas as pd
import subprocess
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
from typing import Dict, Any, Tuple

# [Your existing ML code here]
# Copy all the functions from your script

class MLTrainer:
    @staticmethod
    def train_model_pipeline(dataset_url: str, target_column: str, model_type: str) -> Dict[str, Any]:
        """
        Run the complete ML training pipeline and return results.
        
        Args:
            dataset_url: Kaggle dataset URL
            target_column: Name of the target column
            model_type: Type of model to train
            
        Returns:
            Dict containing training results and metrics
        """
        try:
            dataset_identifier = download_kaggle_dataset(dataset_url)
            data, file_path = load_data(dataset_identifier)
            
            X_train, X_test, y_train, y_test = preprocess_data(data, target_column)
            model = train_model(X_train, y_train, model_type)
            metrics = evaluate_model(model, X_test, y_test)
            
            return {
                "status": "success",
                "metrics": metrics,
                "dataset_info": {
                    "rows": len(data),
                    "columns": list(data.columns),
                    "file_path": file_path
                },
                "model_info": {
                    "type": model_type,
                    "best_params": model.get_params()
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error_message": str(e)
            }
