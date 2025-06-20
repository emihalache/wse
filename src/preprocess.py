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
    
    def country_preprocessing(self, df):
        # pd.set_option("display.max_columns", None)
        # pd.set_option("display.max_rows", None)

        # print(df['country'].unique())
        # Remove rows where country is "Not Given"
        df = df[df['country'] != 'Not Given']

        # The dataset covers 85 countries
        logger.info(f"Dataset before cleaning countries: {df['country'].unique().shape}")
        
        country_counts = df['country'].value_counts()
        # pd.set_option("display.max_rows", None)
        # print(country_counts)

        # adjust here the threshold that defines a significant number of entries
        # per country to be suitable for analysis
        min_entries = 50
        df_signif_count = country_counts[country_counts >= min_entries].index.tolist()
        df = df[df['country'].isin(df_signif_count)]

        # We will analyse 24 countries
        logger.info(f"Dataset after cleaning countries: {df['country'].unique().shape}")
        # print(df['country'].unique())

        return df
    
    def genre_preprocessing(self, df):
        # Map genres to broader categories
        df = self.map_genre_categories(df)
        # print(df['genre_group'].value_counts())

        # But we can still see the distribution within the broad category TV Shows
        # print(df[df['genre_group'] == 'TV Show']['Genre'].value_counts())

        logger.info(f"Dataset after cleaning genres: {df['genre_group'].unique().shape}" )
        # We will analyse 19 genres

        return df
    
    def rating_preprocessing(self, df):
        logger.info(f"Dataset before cleaning maturity ratings: {df['rating'].unique().shape} ")
        # The daaset covers 14 maturity ratings
        
        df = self.map_rating_to_age_appropriateness(df)

        logger.info(f"Dataset after cleaning maturity ratings: {df['age_appropriateness'].unique().shape}")
        # We will analyse 7 age appropriateness categories

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
    
    def map_genre_categories(self, df):
        df = df.copy()
        # Explode genres since the listen_in column contains genres separated by commas
        df['Genre'] = df['listed_in'].str.split(', ')
        df = df.explode('Genre')
        df['Genre'] = df['Genre'].str.strip()  # Remove leading/trailing whitespace

        logger.info(f"Dataset before cleaning genres: {df['Genre'].unique().shape}")
        # The dataset covers 42 genres

        # Remove entries where the genre is just 'Movies'
        df = df[df['Genre'] != 'Movies']

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

        df['genre_group'] = df['Genre'].map(group_genres).fillna('Others')

        return df
