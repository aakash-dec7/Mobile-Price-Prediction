import os
import json
import pickle
import pandas as pd
from Logger import logger
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


class Training:
    def __init__(self):
        """
        Initializes the Training class.
        """
        self.random_state = 42
        self.metrics_dir = "artifacts/Metrics"
        self.model_dir = "artifacts/Model"

        logger.info("Loading training datasets...")
        self.X_train = pd.read_csv("artifacts/Train/X_train.csv").values
        self.y_train = pd.read_csv("artifacts/Train/y_train.csv").values.ravel()
        logger.info(f"X_train.shape: {self.X_train.shape}")
        logger.info(f"y_train.shape: {self.y_train.shape}")

        # Build model pipelines and hyperparameter grids
        self.pipelines = self.build_pipelines()
        self.param_grids = self.get_param_grids()

    def build_pipelines(self):
        """
        Builds preprocessing and model pipelines for each classifier.
        """
        logger.info("Building model pipeline(s)...")
        return {
            "RandomForestClassifier": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        RandomForestClassifier(random_state=self.random_state),
                    ),
                ]
            ),
            "XGBClassifier": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        XGBClassifier(
                            random_state=self.random_state,
                            use_label_encoder=False,
                            eval_metric="logloss",
                        ),
                    ),
                ]
            ),
            "AdaBoostClassifier": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("classifier", AdaBoostClassifier(random_state=self.random_state)),
                ]
            ),
        }

    def get_param_grids(self):
        """
        Defines comprehensive hyperparameter grids for GridSearchCV for each model.
        """
        logger.info("Defining extended hyperparameter grid(s)...")
        return {
            "RandomForestClassifier": {
                "classifier__n_estimators": [30, 50, 100],
                "classifier__max_depth": [10, 15, 20],
                "classifier__min_samples_split": [2, 5],
                "classifier__min_samples_leaf": [1, 2],
                "classifier__max_features": ["sqrt", "log2"],
            },
            "XGBClassifier": {
                "classifier__n_estimators": [30, 50, 100],
                "classifier__max_depth": [3, 6, 9],
                "classifier__learning_rate": [0.01, 0.1, 0.3],
                "classifier__gamma": [0, 1, 5],
                "classifier__reg_lambda": [1, 1.5, 2],  # L2 regularization
            },
            "AdaBoostClassifier": {
                "classifier__n_estimators": [30, 50, 100],
                "classifier__learning_rate": [0.1, 0.5, 1.0],
            },
        }

    def train(self):
        """
        Trains model(s) using GridSearchCV and save the best model and metrics.
        """
        logger.info("Starting training process for all models...")
        metrics_list = []

        for model_name, pipeline in self.pipelines.items():
            logger.info(f"Beginning training for model: {model_name}")
            param_grid = self.param_grids[model_name]

            # Initialize GridSearchCV
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=5,
                scoring="accuracy",
                n_jobs=-1,
                verbose=2,
            )

            # Fit model
            grid_search.fit(self.X_train, self.y_train)
            logger.info(f"{model_name} training complete.")
            logger.info(f"Best accuracy score: {grid_search.best_score_:.4f}")
            logger.info(f"Optimal parameters: {grid_search.best_params_}")

            # Record training metrics
            metrics = {
                "model": model_name,
                "best_params": grid_search.best_params_,
                "best_score": grid_search.best_score_,
            }
            metrics_list.append(metrics)

        # Save best model
        logger.info("Saving the best model to disk...")
        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, "model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(grid_search.best_estimator_, f)
        logger.info(f"Model saved successfully at: {model_path}")

        # Save metrics to JSON
        logger.info("Saving training metrics to disk...")
        os.makedirs(self.metrics_dir, exist_ok=True)
        metrics_path = os.path.join(self.metrics_dir, "train_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics_list, f, indent=4)
        logger.info(f"Training metrics saved at: {metrics_path}")
        logger.info("Training process completed for all models.")


if __name__ == "__main__":
    try:
        logger.info("Starting Training pipeline...")
        trainer = Training()
        trainer.train()
        logger.info("Training pipeline completed successfully.")
    except Exception as e:
        logger.error("Training pipeline failed.", exc_info=True)
        raise RuntimeError("Training pipeline failed.") from e
