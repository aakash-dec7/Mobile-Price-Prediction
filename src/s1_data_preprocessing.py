import os
import json
import pandas as pd
from Logger import logger


class DataPreprocessing:
    def __init__(self):
        """
        Initializes the DataPreprocessing class.
        """
        self.save_dataset_dir = "artifacts/Preprocessed"

        # Load the dataset
        self.df = pd.read_csv("artifacts/dataset.csv")
        logger.info(f"Dataset loaded successfully with shape: {self.df.shape}")

    def drop_na(self):
        """
        Drops rows with missing values from the dataset.
        """
        logger.info("Dropping rows with missing values...")
        self.df = self.df.dropna()

    def drop_cols(self):
        """
        Drops unnecessary columns and separates features (X) and target (y).
        """
        logger.info("Dropping unnecessary columns and splitting features/target...")

        columns_to_drop = ["mobile_wt", "m_dep", "talk_time", "price_range"]

        self.y = self.df["price_range"]
        self.X = self.df.drop(columns_to_drop, axis=1)

    def save_df(self):
        """
        Saves the processed feature set (X) and target variable (y) as CSV files.
        """
        logger.info("Saving preprocessed features and target to disk...")

        os.makedirs(self.save_dataset_dir, exist_ok=True)

        # Construct paths for saving
        X_path = os.path.join(self.save_dataset_dir, "X.csv")
        y_path = os.path.join(self.save_dataset_dir, "y.csv")

        # Save to CSV
        self.X.to_csv(X_path, index=False)
        self.y.to_csv(y_path, index=False)

        logger.info(f"Features saved to: {X_path}")
        logger.info(f"Target saved to: {y_path}")
        logger.info("All preprocessed files saved successfully.")

    def run(self):
        """
        Executes the preprocessing pipeline.
        """
        self.drop_na()
        self.drop_cols()
        self.save_df()


if __name__ == "__main__":
    try:
        logger.info("Starting DataPreprocessing pipeline...")
        data_preprocessing = DataPreprocessing()
        data_preprocessing.run()
        logger.info("DataPreprocessing pipeline completed successfully.")
    except Exception as e:
        logger.error("DataPreprocessing pipeline failed.", exc_info=True)
        raise RuntimeError("DataPreprocessing pipeline failed.") from e
