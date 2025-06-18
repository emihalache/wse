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


    def country_preprocessing(self, df):
        # pd.set_option("display.max_columns", None)
        # pd.set_option("display.max_rows", None)

        # print(df['country'].unique())
        # Remove rows where country is "Not Given"
        df = df[df['country'] != 'Not Given']

        # The dataset covers 85 countries
        print("Dataset before cleaning countries: ", df['country'].unique().shape)
        
        country_counts = df['country'].value_counts()
        # pd.set_option("display.max_rows", None)
        # print(country_counts)

        # adjust here the threshold that defines a significant number of entries
        # per country to be suitable for analysis
        min_entries = 50
        df_signif_count = country_counts[country_counts >= min_entries].index.tolist()
        df = df[df['country'].isin(df_signif_count)]

        # We will analyse 24 countries
        print("Dataset after cleaning countries: ", df['country'].unique().shape)
        # print(df['country'].unique())

        return df
    

    def map_genre_categories(self, df):
        # Explode genres since the listen_in column contains genres separated by commas
        df_genre = df.copy()
        df_genre['Genre'] = df_genre['listed_in'].str.split(', ')
        df_genre = df_genre.explode('Genre')

        # Remove entries where the genre is just 'Movies'
        df_genre = df_genre[df_genre['Genre'] != 'Movies']

        # Mapping genres to their broader categories,
        # according to Netflix codes https://www.netflix-codes.com/
        group_genres = {     
            'Action & Adventure': 'Action & adventure',

            'Anime Features': 'Anime',
            'Anime Series': 'Anime',

            'Children & Family Movies': 'Children & family movies',

            'Classic Movies': 'Classic Movies',

            'Comedies': 'Comedies',
            'Stand-Up Comedy': 'Comedies',
            'Stand-Up Comedy & Talk Shows': 'Comedies',

            'Documentaries': 'Documentaries',
            'Docuseries': 'Documentaries',

            'Dramas': 'Dramas',

            'International Movies': 'Foreign movies',

            'Horror Movies': 'Horror movies',

            'Independent Movies': 'Independent movies',

            'LGBTQ Movies': 'LGBTQ+',

            'Music & Musicals': 'Music',

            'Romantic Movies': 'Romantic movies',

            'Sci-Fi & Fantasy': 'Sci - Fi & Fantasy',

            'Sports Movies': 'Sports movies',
            
            'Crime TV Shows': 'TV Show',
            'TV Action & Adventure': 'TV Show',
            'TV Dramas': 'TV Show',
            'TV Horror': 'TV Show',
            'TV Mysteries': 'TV Show',
            'British TV Shows': 'TV Show',
            'Reality TV': 'TV Show',
            'Kids\' TV': 'TV Show',
            'TV Comedies': 'TV Show',
            'Korean TV Shows': 'TV Show',
            'Science & Nature TV': 'TV Show',
            'TV Shows': 'TV Show',
            'International TV Shows': 'TV Show',
            'Spanish-Language TV Shows': 'TV Show',
            'TV Thrillers': 'TV Show',
            'Romantic TV Shows': 'TV Show',
            'TV Sci-Fi & Fantasy': 'TV Show',
            'Classic & Cult TV': 'TV Show',
            
            'Thrillers': 'Thrillers',

            'Teen TV Shows': 'Teen TV shows',

            'Faith & Spirituality': 'Others',
            'Cult Movies': 'Others',
        }

        df_genre['genre_group'] = df_genre['Genre'].map(group_genres).fillna('Others')

        return df_genre

    def genre_preprocessing(self, df):
        # Explode genres since the listen_in column contains genres separated by commas
        df['Genre'] = df['listed_in'].str.split(', ')
        df = df.explode('Genre')

        print("Dataset before cleaning genres: ", df['Genre'].unique().shape)
        # The dataset covers 42 genres

        # Map genres to broader categories
        df = self.map_genre_categories(df)
        # print(df['genre_group'].value_counts())

        # But we can still see the distribution within the broad category TV Shows
        # print(df[df['genre_group'] == 'TV Show']['Genre'].value_counts())

        print("Dataset after cleaning genres: ", df['genre_group'].unique().shape)

        return df
    

    def map_rating_to_age_appropriateness(self, df):
        # Mapping maturity ratings to their broader categories, because:
        # - Some categories only differ by terminology used in movies vs TV shows
        # - Our analysis is focused on the age appropriateness of content, not the reasons why 
        # a system is rated a certain way (e.g. including violence, language, etc.)
        group_maturity_ratings = {
            # Intended or restricted to mature audiences and not suitable for children under 17
            'TV-MA': 'Adult (17+)',
            'R': 'Adult (17+)',
            'NC-17': 'Adult (17+)',
            
            # Some material may be inappropriate for children under 13 or 14 years old, so parents 
            # are strongly urged to be cautious of the content
            'TV-14': 'Teens (PG Strongly Cautioned)',
            'PG-13': 'Teens (PG Strongly Cautioned)',
            
            # Some material may not be suitable for young children, so parental guidance is suggested.
            'TV-PG': 'Young Children (PG Suggested)',
            'PG': 'Young Children (PG Suggested)',
            
            # The content is intended for older children (age 7 and above)
            'TV-Y7': 'Older Children (7+)',
            'TV-Y7-FV': 'Older Children (7+)',
            
            # The content is designed to be appropriate for children of all ages
            'TV-Y': 'Young Children',
            
            # The content is suitable for all ages (general audience), including young children
            'TV-G': 'All Ages',
            'G': 'All Ages',
            
            # The content has not been assigned a specific rating, thus viewers should use their discretion
            'NR': 'Not Rated',
            'UR': 'Not Rated'
        }

        df['age_appropriateness'] = df['rating'].map(group_maturity_ratings).fillna('Other')

        return df  
    
    def rating_preprocessing(self, df):
        print("Dataset before cleaning maturity ratings: ", df['rating'].unique().shape)
        
        df = self.map_rating_to_age_appropriateness(df)

        print("Dataset after cleaning maturity ratings: ", df['age_appropriateness'].unique().shape)

        return df
    

    def type_by_country_visualizations(self, df):
        # Grouping releases by country and content type
        releases_by_country = df.groupby(['country', 'type']).size().unstack(fill_value=0)

        # Order by the number of releases
        ordered_index = releases_by_country.sum(axis=1).sort_values(ascending=False).index

        # -----------------------------
        # Plot 1: Quantity of releases per country with content type shown
        # -----------------------------
        type_by_country = releases_by_country.loc[ordered_index]

        type_by_country.plot(kind='bar', stacked=True, figsize=(10, 5), colormap='tab20')
        plt.title("Releases by Country")
        plt.xlabel("Country")
        plt.ylabel("Number of Releases")
        plt.legend(title='Content Type')
        plt.tight_layout()
        plt.savefig("results/s2_1_releases_by_country.png")
        plt.clf()  # Clear figure for next plot

        # -----------------------------
        # Plot 2: Proportions of type of content released per country
        # -----------------------------
        
        # Normalize to get proportions of content type per country
        normalized = releases_by_country.div(releases_by_country.sum(axis=1), axis=0)

        type_proportions_by_country = normalized.loc[ordered_index]

        type_proportions_by_country.plot(kind='bar', stacked=True, figsize=(10, 5), colormap='tab20')
        plt.title("Proportion of Content Type by Country")
        plt.xlabel("Country")
        plt.ylabel("Percentage")
        plt.legend(title='Content Type', bbox_to_anchor=(1, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig("results/s2_2_types_by_country.png")
        plt.clf()  # Clear figure for next plot


    def ratings_by_country_visualizations(self, df):
        # -----------------------------
        # Plot 3: Maturity Rating Proportions by Country
        # -----------------------------

        # Grouping releases by country and content type
        rating_by_country = df.groupby(['country', 'rating']).size().unstack(fill_value=0)

        # Normalize to get proportions of maturity ratings per country
        normalized = rating_by_country.div(rating_by_country.sum(axis=1), axis=0)

        # Order by the number of releases
        ordered_index = normalized.loc[rating_by_country.sum(axis=1).sort_values(ascending=False).index]

        # Order rating categories by average size across all countries for better visualization
        # So generally the smallest one is at the bottom, and largest is at the top
        rating_proportions_by_country = ordered_index[ordered_index.mean().sort_values(ascending=True).index]

        rating_proportions_by_country.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='tab20')
        plt.title("Normalized Maturity Ratings by Country")
        plt.xlabel("Country")
        plt.ylabel("Percentage")
        plt.legend(title='Maturity Rating', bbox_to_anchor=(1, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig("results/s2_3_normalized_ratings_by_country.png")
        plt.clf() # Clear figure for next plot

        # -----------------------------
        # Plot 4: Proportions of Age Appropriateness Content by Country
        # -----------------------------

        # Grouping releases by country and content type
        grouped_rating_by_country = df.groupby(['country', 'age_appropriateness']).size().unstack(fill_value=0)

        # Normalize to get proportions of maturity ratings per country
        normalized = grouped_rating_by_country.div(grouped_rating_by_country.sum(axis=1), axis=0)

        # Order by the number of releases
        ordered_index = normalized.loc[grouped_rating_by_country.sum(axis=1).sort_values(ascending=False).index]

        # Order rating categories by average size across all countries for better visualization
        # So generally the smallest one is at the bottom, and largest is at the top
        grouped_rating_proportions_by_country = ordered_index[ordered_index.mean().sort_values(ascending=True).index]

        grouped_rating_proportions_by_country.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='tab20')
        plt.title("Normalized Age Appropriateness by Country")
        plt.xlabel("Country")
        plt.ylabel("Percentage")
        plt.legend(title='Age Appropriateness', bbox_to_anchor=(1, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig("results/s2_4_normalized_grouped_ratings_by_country.png")
        plt.clf() # Clear figure for next plot


    def genres_by_country_visualizations(self, df):
        # -----------------------------
        # Plot 5: Genre by Country Heatmap
        # -----------------------------

        # Grouping releases by country and genre
        genre_by_country = df.groupby(['country', 'Genre']).size().unstack(fill_value=0)

        # Order by the number of releases
        ordered_index = genre_by_country.sum(axis=1).sort_values(ascending=False).index

        genre_by_country = genre_by_country.loc[ordered_index]

        plt.figure(figsize=(14, 8))
        sns.heatmap(genre_by_country, annot=False, cmap='viridis')
        plt.title("Genres by Country")
        plt.xlabel("Genre")
        plt.ylabel("Country")
        plt.tight_layout()
        plt.savefig("results/s2_5_genres_by_country.png")
        plt.clf() # Clear figure for next plot
        
        # -----------------------------
        # Plot 6: Genre Groups by Country Heatmap
        # -----------------------------

        # Grouping releases by country and genre group
        genre_group_by_country = df.groupby(['country', 'genre_group']).size().unstack(fill_value=0)

        # Order by the number of releases
        ordered_index = genre_group_by_country.sum(axis=1).sort_values(ascending=False).index

        genre_group_by_country = genre_group_by_country.loc[ordered_index]

        plt.figure(figsize=(14, 8))
        sns.heatmap(genre_group_by_country, annot=False, cmap='viridis')
        plt.title("Genre Groups by Country")
        plt.xlabel("Genre Group")
        plt.ylabel("Country")
        plt.tight_layout()
        plt.savefig("results/s2_6_genre_groups_by_country.png")
        plt.clf() # Clear figure for next plot

        # -----------------------------
        # Plot 7: TV Show Sub-Genre by Country Heatmap
        # -----------------------------

        # Filter for TV Shows
        df_tv_show = df[df['genre_group'] == 'TV Show']
        # Count (Genre, Country) pairs
        tv_show_subgenre_by_country = pd.crosstab(df_tv_show['Genre'], df_tv_show['country'])
        
        plt.figure(figsize=(14, 8))
        sns.heatmap(tv_show_subgenre_by_country, annot=False, cmap='viridis')
        plt.title('TV Show Sub-genres by Country')
        plt.xlabel('Country')
        plt.ylabel('TV Show Sub-Genre')
        plt.tight_layout()
        plt.savefig("results/s2_7_tv_show_subgenres_by_country.png")
        plt.clf() # Clear figure for next plot
        

    def sub2(self, df):
        print("****************************** S2: Regional Variations in Content Characteristics ******************************")

        # Ensure results directory exists
        os.makedirs("results", exist_ok=True)

        ''' Data Cleaning and Preparation '''
        df = self.country_preprocessing(df)
        df = self.genre_preprocessing(df)
        df = self.rating_preprocessing(df)

        ''' Visualizations of Content Type by Country '''
        self.type_by_country_visualizations(df)

        ''' Visualizations of Maturity Ratings by Country '''
        self.ratings_by_country_visualizations(df)
        
        ''' Visualizations of Genres by Country '''
        self.genres_by_country_visualizations(df)
