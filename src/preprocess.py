import pandas as pd
import logging

#region logging stuff
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
#endregion

class Preprocessor:
    def preprocess(self, df):
        subset_cols = [col for col in df.columns if col != 'id']
        new_df = df.drop_duplicates(subset=subset_cols, ignore_index=True)

        # The amount of data that have null values
        logger.info(f"\n{new_df.isnull().sum()}")

        # We know that there are no null_values

        return new_df