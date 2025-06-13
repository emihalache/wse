import pandas as pd
import matplotlib.pyplot as plt
import logging
import os
import seaborn as sns

from distinctipy import distinctipy

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
    def sub1(self, df):
        # Ensure results directory exists
        os.makedirs("results", exist_ok=True)

        if 'release_year' not in df.columns:
            print("release_year column not found.")
            return
        
        # Convert release_year to datetime
        df['Date_N'] = pd.to_datetime(df['release_year'].astype(int).astype(str) + '-01-01')
        df['Year'] = df['Date_N'].dt.year

        # -----------------------------
        # Plot 1: Total releases per year
        # -----------------------------
        releases_per_year = df['Year'].value_counts().sort_index()
        releases_per_year.plot(kind='bar', figsize=(10, 5))
        plt.title("Number of Releases by Year")
        plt.xlabel("Year")
        plt.ylabel("Number of Shows")
        plt.tight_layout()
        plt.savefig("results/releases_by_year.png")
        plt.clf()  # Clear figure for next plot

        # Save data to Excel
        releases_df = releases_per_year.reset_index()
        releases_df.columns = ['Year', 'Count']
        releases_df.to_excel("results/releases_by_year.xlsx", index=False)


        # -----------------------------
        # Plot 2: Movies per genre per year
        # -----------------------------
        if 'listed_in' not in df.columns:
            print("listed_in column not found.")
            return
        
        # Split the 'listed_in' column by comma and explode
        df['Genre'] = df['listed_in'].str.split(', ')
        df_exploded = df.explode('Genre')

        genre_counts = df_exploded.groupby(['Year', 'Genre']).size().unstack(fill_value=0)
        # Generate visually distinct colors
        distinct_colors = distinctipy.get_colors(genre_counts.shape[1])

        # Plot manually to control linestyle
        plt.figure(figsize=(20, 10))
        linestyles = ['-', ':']  # Alternating solid and dotted lines

        for i, (genre, color) in enumerate(zip(genre_counts.columns, distinct_colors)):
            linestyle = linestyles[i % len(linestyles)]
            plt.plot(genre_counts.index, genre_counts[genre], label=genre, color=color, linestyle=linestyle)


        plt.title("Number of Movies per Genre per Year")
        plt.xlabel("Year")
        plt.ylabel("Count")
        plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig("results/movies_per_genre_per_year.png")
        plt.clf()

        # Save genre data to Excel
        genre_counts.to_excel("results/movies_per_genre_per_year.xlsx")

        # -----------------------------
        # Plot 3: Movies/Series per type per year
        # -----------------------------
        if 'type' not in df.columns:
            print("type column not found.")
            return
        # Group by Year and Type
        type_counts = df.groupby(['Year', 'type']).size().unstack(fill_value=0)

        # Plot as bar chart
        type_counts.plot(kind='bar', stacked=True, figsize=(12, 6))
        plt.title("Number of Movies and TV Shows per Type per Year")
        plt.xlabel("Year")
        plt.ylabel("Count")
        plt.legend(title='Type')
        plt.tight_layout()
        plt.savefig("results/types_per_year.png")
        plt.clf()

        # Save to Excel
        type_counts.to_excel("results/types_per_year.xlsx")

        # -----------------------------
        # Plot 4: Movies/Series per rating per year
        # -----------------------------
        if 'rating' not in df.columns:
            print("rating column not found.")
            return
        # Group by Year and Rating
        rating_counts = df.groupby(['Year', 'rating']).size().unstack(fill_value=0)

        # Generate visually distinct colors
        distinct_colors_type = distinctipy.get_colors(rating_counts.shape[1])

        rating_counts.plot(
            kind='line',
            stacked=False,
            figsize=(20, 10),
            color=distinct_colors_type,
        )

        plt.title("Number of Movies and TV Shows per Rating per Year")
        plt.xlabel("Year")
        plt.ylabel("Count")
        plt.legend(title='Rating', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig("results/ratings_per_year.png")
        plt.clf()

        # Save to Excel
        rating_counts.to_excel("results/ratings_per_year.xlsx")

        
        # -----------------------------
        # Plot 5: Top 10 Directors per Year
        # -----------------------------
        # if 'director' not in df.columns:
        #     logger.warning("'director' column not found.")
        #     return

        # # Clean up
        # df['Director'] = df['director'].str.strip()

        # # Count appearances per director per year
        # director_counts = df.groupby(['Year', 'Director']).size().reset_index(name='Count')

        # # Get top 10 directors per year
        # top10_per_year = director_counts.groupby('Year').apply(
        #     lambda g: g.nlargest(10, 'Count')
        # ).reset_index(drop=True)

        # # Pivot to wide format for plotting
        # pivot_df = top10_per_year.pivot(index='Year', columns='Director', values='Count').fillna(0)

        # # Generate distinct colors
        # distinct_colors_directors = distinctipy.get_colors(pivot_df.shape[1])

        # # Plot
        # pivot_df.plot(
        #     kind='bar',
        #     stacked=True,
        #     figsize=(20, 10),
        #     color=distinct_colors_directors,
        # )

        # plt.title("Top 10 Directors per Year")
        # plt.xlabel("Year")
        # plt.ylabel("Number of Movies/Shows")
        # plt.legend(title="Director", bbox_to_anchor=(1.05, 1), loc='upper left')
        # plt.tight_layout()
        # plt.savefig("results/top10_directors_per_year.png")
        # plt.clf()

        # # Save to Excel
        # pivot_df.to_excel("results/top10_directors_per_year.xlsx")


    def genre(self, df):
        if 'listed_in' not in df.columns:
            print("listed_in column not found.")
            return

        # Skip rows with missing release_year
        df = df[df['listed_in'].notnull()]

    def sub2(self, df):
        # pd.set_option("display.max_columns", None)
        # pd.set_option("display.max_rows", None)

        # Ensure results directory exists
        os.makedirs("results", exist_ok=True)

        # print(df['country'].unique())
        # Remove rows where country is "Not Given"
        df = df[df['country'] != 'Not Given']

        print("Before cleaning: ", df['country'].unique().shape)
        # The dataset covers 85 countries

        country_counts = df['country'].value_counts()
        # pd.set_option("display.max_rows", None)
        # print(country_counts)

        # adjust the threshold for significant countries for analysis
        min_entries = 50
        df_signif_count = country_counts[country_counts >= min_entries].index.tolist()
        df = df[df['country'].isin(df_signif_count)]

        # We will analyse 24 countries
        print("After cleaning: ", df['country'].unique().shape)

        # -----------------------------
        # Plot 1: Content Type by Country
        # -----------------------------
        releases_by_type = df.groupby(['country', 'type']).size().unstack(fill_value=0)
        type_by_country = releases_by_type.loc[releases_by_type.sum(axis=1).sort_values(ascending=False).index]
        # releases_type_per_country = releases_by_type['country'].value_counts().sort_values(ascending=False)
        # ordered by the number of releases
        type_by_country.plot(kind='bar', stacked=True, figsize=(10, 5), colormap='tab20')
        plt.title("Content Type by Country")
        plt.xlabel("Country")
        plt.ylabel("Number of Shows")
        plt.tight_layout()
        plt.savefig("results/types_by_country.png")
        plt.clf()  # Clear figure for next plot

        # -----------------------------
        # Plot 2: Maturity Rating by Country
        # -----------------------------
        rating_counts = df.groupby(['country', 'rating']).size().unstack(fill_value=0)
        normalized = rating_counts.div(rating_counts.sum(axis=1), axis=0)
        # ordered by the number of releases
        normalized = normalized.loc[rating_counts.sum(axis=1).sort_values(ascending=False).index]

        # Calculate average proportions per rating
        avg_props = normalized.mean().sort_values(ascending=True)  # smallest at bottom, largest at top
        # Reorder columns of normalized dataframe accordingly
        normalized = normalized[avg_props.index]

        normalized.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='tab20')
        plt.title("Normalized Maturity Ratings by Country")
        plt.xlabel("Country")
        plt.ylabel("Proportion of Shows")
        plt.legend(title='Maturity Rating', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig("results/normalized_ratings_by_country.png")
        plt.clf()

        # -----------------------------
        # Plot 3: Genre by Country Heatmap
        # -----------------------------
        # Explode genres
        df_genre = df.copy()
        df_genre['Genre'] = df_genre['listed_in'].str.split(', ')
        df_genre = df_genre.explode('Genre')

        # Group and plot
        genre_by_country = df_genre.groupby(['country', 'Genre']).size().unstack(fill_value=0)
        genre_by_country = genre_by_country.loc[genre_by_country.sum(axis=1).sort_values(ascending=False).index]

        plt.figure(figsize=(14, 8))
        sns.heatmap(genre_by_country, annot=False, cmap='viridis')
        plt.title("Genres by Country (Heatmap)")
        plt.xlabel("Genre")
        plt.ylabel("Country")
        plt.tight_layout()
        plt.savefig("results/genres_by_country.png")
        plt.clf()
        