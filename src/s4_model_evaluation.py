import os
import json
import pickle
import pandas as pd
from Logger import logger
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class Evaluation:
    def __init__(self):
        """
        Initializes Evaluation class.
        """
        self.model_path = "artifacts/Model/model.pkl"
        self.metrics_path = "artifacts/Metrics"

        # Load the trained model from disk
        logger.info(f"Loading trained model from: {self.model_path}")
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
        logger.info("Model loaded successfully.")

        # Load test features and labels
        logger.info("Loading test datasets from CSV files.")
        self.X_test = pd.read_csv("artifacts/Test/X_test.csv").values
        self.y_test = pd.read_csv("artifacts/Test/y_test.csv").values.ravel()
        logger.info(f"X_test.shape: {self.X_test.shape}")
        logger.info(f"y_test.shape: {self.y_test.shape}")

    def eval(self):
        """
        Evaluates the model on the test dataset.
        """
        logger.info("Starting evaluation of the model...")

        # Predict on test set
        logger.info("Generating predictions on test data...")
        y_pred = self.model.predict(self.X_test)
        logger.info("Predictions generated successfully.")

        # Calculate performance metrics
        logger.info("Calculating evaluation metrics...")
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(
            self.y_test, y_pred, average="weighted", zero_division=0
        )
        recall = recall_score(self.y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(self.y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(self.y_test, y_pred).tolist()
        report = classification_report(
            self.y_test, y_pred, zero_division=0, output_dict=True
        )
        logger.info("Evaluation metrics calculated.")

        # Ensure metrics directory exists
        os.makedirs(self.metrics_path, exist_ok=True)
        eval_metrics_path = os.path.join(self.metrics_path, "eval_metrics.json")

        # Compile all metrics into a dictionary
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm,
            "classification_report": report,
        }

        # Save metrics dictionary as a formatted JSON file
        logger.info(f"Saving evaluation metrics to {eval_metrics_path}")
        with open(eval_metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info("Evaluation metrics saved successfully.")


if __name__ == "__main__":
    try:
        logger.info("Starting Evaluation pipeline...")
        evaluator = Evaluation()
        evaluator.eval()
        logger.info("Evaluation pipeline completed successfully.")
    except Exception as e:
        logger.error("Evaluation pipeline failed.", exc_info=True)
        raise RuntimeError("Evaluation pipeline failed.") from e
