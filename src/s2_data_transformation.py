import os
import pandas as pd
from Logger import logger
from sklearn.model_selection import train_test_split


class DataTransformation:
    def __init__(self):
        """
        Initializes the DataTransformation class.
        """
        # Load preprocessed X and y CSV datasets
        self.X = pd.read_csv("artifacts/Preprocessed/X.csv")
        self.y = pd.read_csv("artifacts/Preprocessed/y.csv")

        logger.info(f"X.shape: {self.X.shape}")
        logger.info(f"y.shape: {self.y.shape}")

        # Output directory paths
        self.save_train_data_path = "artifacts/Train"
        self.save_test_data_path = "artifacts/Test"

        # Split configuration
        self.test_size = 0.3
        self.random_state = 42

    def split_data(self):
        """
        Splits the data into training and testing sets and saves them to disk.
        """
        logger.info("Starting data split...")

        # Perform train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )

        logger.info(
            f"Data split completed successfully. "
            f"Training samples: {len(X_train)}, Test samples: {len(X_test)}"
        )

        # Ensure output directories exist
        os.makedirs(self.save_train_data_path, exist_ok=True)
        os.makedirs(self.save_test_data_path, exist_ok=True)

        # Save training data
        X_train.to_csv(
            os.path.join(self.save_train_data_path, "X_train.csv"), index=False
        )
        y_train.to_csv(
            os.path.join(self.save_train_data_path, "y_train.csv"), index=False
        )
        logger.info("Training data saved successfully to artifacts/Train/")

        # Save test data
        X_test.to_csv(os.path.join(self.save_test_data_path, "X_test.csv"), index=False)
        y_test.to_csv(os.path.join(self.save_test_data_path, "y_test.csv"), index=False)
        logger.info("Test data saved successfully to artifacts/Test/")

    def run(self):
        """
        Executes the data transformation pipeline.
        """
        self.split_data()


if __name__ == "__main__":
    try:
        logger.info("Starting DataTransformation pipeline...")
        data_transformation = DataTransformation()
        data_transformation.run()
        logger.info("DataTransformation pipeline completed successfully.")
    except Exception as e:
        logger.error("DataTransformation pipeline failed.", exc_info=True)
        raise RuntimeError("DataTransformation pipeline failed.") from e
