import pandas as pd
import logging
import os
from src.dataset import Retriever
from src.preprocess import Preprocessor

# region logging stuff
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


# endregion

def extract_genres_to_csv():
    """
    Extract all unique genres from the Netflix dataset and save to CSV
    """
    # Initialize classes
    retriever = Retriever()
    preprocessor = Preprocessor()

    # Get and preprocess the dataset
    df = retriever.get_dataset()
    if df is None:
        logger.error("Failed to retrieve dataset")
        return

    df = preprocessor.preprocess(df)
    logger.info(f"Dataset shape: {df.shape}")

    # Check if 'listed_in' column exists
    if 'listed_in' not in df.columns:
        logger.error("'listed_in' column not found in dataset")
        return

    # Extract genres
    logger.info("Extracting genres from 'listed_in' column...")

    # Remove rows with null values in listed_in
    df_clean = df[df['listed_in'].notnull()].copy()

    # Split the genres and create a list of all genres
    all_genres = []
    for genres_string in df_clean['listed_in']:
        # Split by comma and strip whitespace
        genres = [genre.strip() for genre in genres_string.split(',')]
        all_genres.extend(genres)

    # Get unique genres and sort them
    unique_genres = sorted(list(set(all_genres)))

    # Count occurrences of each genre
    genre_counts = {}
    for genre in all_genres:
        genre_counts[genre] = genre_counts.get(genre, 0) + 1

    # Create DataFrame with genre information
    genre_df = pd.DataFrame([
        {
            'genre': genre,
            'count': genre_counts[genre],
            'percentage': round((genre_counts[genre] / len(df_clean)) * 100, 2)
        }
        for genre in unique_genres
    ])

    # Sort by count (descending)
    genre_df = genre_df.sort_values('count', ascending=False).reset_index(drop=True)

    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)

    # Save to CSV
    output_file = "results/all_genres.csv"
    genre_df.to_csv(output_file, index=False)

    # Log results
    logger.info(f"Found {len(unique_genres)} unique genres")
    logger.info(f"Total genre entries: {len(all_genres)}")
    logger.info(f"Results saved to: {output_file}")

    # Print top 10 genres
    print("\n=== TOP 10 GENRES ===")
    print(genre_df.head(10).to_string(index=False))

    # Print some statistics
    print(f"\nTotal unique genres: {len(unique_genres)}")
    print(f"Most common genre: {genre_df.iloc[0]['genre']} ({genre_df.iloc[0]['count']} occurrences)")
    print(f"Least common genres: {len(genre_df[genre_df['count'] == 1])} genres appear only once")

    return genre_df

def extract_countries_to_csv():
    """
    Extract all unique countries from the Netflix dataset and save to CSV
    """
    # Initialize classes
    retriever = Retriever()
    preprocessor = Preprocessor()

    # Get and preprocess the dataset
    df = retriever.get_dataset()
    if df is None:
        logger.error("Failed to retrieve dataset")
        return

    df = preprocessor.preprocess(df)
    logger.info(f"Dataset shape: {df.shape}")

    # Check if 'country' column exists
    if 'country' not in df.columns:
        logger.error("'country' column not found in dataset")
        return

    # Extract countries
    logger.info("Extracting countries from 'country' column...")

    # Remove rows with null values in country
    df_clean = df[df['country'].notnull()].copy()

    # Get unique countries and their counts
    country_counts = df_clean['country'].value_counts().reset_index()
    country_counts.columns = ['country', 'count']

    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)

    # Save to CSV
    output_file = "results/all_countries.csv"
    country_counts.to_csv(output_file, index=False)

    # Log results
    logger.info(f"Found {len(country_counts)} unique countries")
    logger.info(f"Results saved to: {output_file}")


if __name__ == '__main__':
    extract_genres_to_csv()
    extract_countries_to_csv()