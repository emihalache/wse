import pandas as pd
import matplotlib.pyplot as plt
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

class Analyzer:
    def year(self, df):
        if 'release_year' not in df.columns:
            print("release_year column not found.")
            return

        # Skip rows with missing release_year
        df = df[df['release_year'].notnull()]

        # Convert release_year (int or float) to string + dummy date
        df['Date_N'] = pd.to_datetime(df['release_year'].astype(int).astype(str) + '-01-01')

        # Plot number of shows released per year
        df['Date_N'].dt.year.value_counts().sort_index().plot(kind='bar', figsize=(10, 5))
        plt.title("Number of Releases by Year")
        plt.xlabel("Year")
        plt.ylabel("Number of Shows")
        plt.tight_layout()
        # Save the plot to results map
        plt.savefig("results/releases_by_year.png")

    def genre(self, df):
        if 'listed_in' not in df.columns:
            print("listed_in column not found.")
            return

        # Skip rows with missing release_year
        df = df[df['listed_in'].notnull()]


