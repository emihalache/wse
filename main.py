import logging

from src.dataset import Retriever
from src.perprocess import Preprocessor

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

if __name__ == '__main__':
    # Initialise all the necessary classes
    retriever = Retriever()
    preprocessor = Preprocessor()

    # Retrieve the dataset as a pandas df
    df = retriever.get_dataset()
    logger.info(f"Shape of df: {df.shape}")
    
    # Start preprocessing the data
    new_df = preprocessor.preprocess(df)
    logger.info(f"New shape of the df: {new_df.shape}")
    