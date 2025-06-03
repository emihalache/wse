import pandas as pd
import kagglehub
import logging
import os

# Set up logging to file and console
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler('myapp.log')
file_handler.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Common format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers if not already added (avoid duplicate logs on rerun)
if not logger.hasHandlers():
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

class Retriever:
    def get_dataset(self):
        # Download latest version of netflix dataset
        path = kagglehub.dataset_download("shivamb/netflix-shows")

        logger.info(f"Path to dataset files: {path}")

        # Look for a CSV file in the dataset folder
        for file_name in os.listdir(path):
            if file_name.endswith('.csv'):
                full_path = os.path.join(path, file_name)
                logger.info(f"Found CSV file: {full_path}")

                # Load CSV as a pandas DataFrame
                df = pd.read_csv(full_path)
                return df  

        logger.warning("No CSV file found in the dataset folder.")
        return None


