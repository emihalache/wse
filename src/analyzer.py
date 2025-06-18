import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import os
from adjustText import adjust_text
from scipy.stats import f_oneway



from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

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

    def sub3(self, df):
        os.makedirs("results", exist_ok=True)
        logger.info("Starting sub3 unsupervised analysis")

        df = df.loc[
            df['country']
            .notna()
            .str.strip()
            .str.lower()
            .ne('not given')
        ]

        tmp = df[['release_year', 'listed_in']].dropna()
        tmp['genre_list'] = tmp['listed_in'].str.split(',\s*')
        tmp = tmp.explode('genre_list')
        year_genre = tmp.groupby(['release_year', 'genre_list']).size().unstack(fill_value=0)
        year_genre_norm = year_genre.div(year_genre.sum(axis=1), axis=0)

        # choose best k for years
        best_k_years = self._choose_k(year_genre_norm, k_min=2, k_max=15, tag="years")
        best_k_years = 4

        pca = PCA(n_components=2, random_state=0)
        year_coords = pca.fit_transform(year_genre_norm)
        kmeans_y = KMeans(n_clusters=best_k_years, random_state=0).fit(year_coords)
        year_labels = kmeans_y.labels_

        plt.figure(figsize=(10, 8))
        colors = distinctipy.get_colors(best_k_years)
        for c in range(best_k_years):
            mask = (year_labels == c)
            plt.scatter(year_coords[mask, 0], year_coords[mask, 1],
                        color=colors[c], label=f'Cluster {c}')

        texts = []
        for idx, yr in enumerate(year_genre_norm.index):
            x, y = year_coords[idx]
            texts.append(
                plt.text(x, y, str(int(yr)),
                         fontsize=8, alpha=0.75)
            )

        adjust_text(
            texts,
            only_move={'points': 'xy', 'texts': 'xy'},
            force_text=(0.5, 0.5),
            force_points=(0.5, 0.5),
            expand_text=(1.2, 1.2),
            expand_points=(1.2, 1.2),
            arrowprops=dict(arrowstyle='-', color='black', lw=0.5, alpha=0.5)
        )

        plt.title("Temporal Clusters of Years by Genre Distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig("results/sub3_temporal_clusters.png")
        plt.close()

        pd.DataFrame({
            'year': year_genre_norm.index,
            'cluster': year_labels
        }).to_excel("results/temporal_clusters.xlsx", index=False)

        # --- ANOVA on each genre across year‐clusters ---
        print("\n=== ANOVA: Temporal clusters ===")
        for genre in year_genre_norm.columns:
            groups = [year_genre_norm[year_labels == c][genre] for c in np.unique(year_labels)]
            stat, p = f_oneway(*groups)
            if p < 0.01:
                print(f"{genre:20s} p={p:.2e}")

        year_centroids = (
            year_genre_norm
            .assign(cluster=year_labels)
            .groupby('cluster')
            .mean()
        )
        # Print top 5 genres per cluster
        for c, row in year_centroids.iterrows():
            top5 = row.sort_values(ascending=False).head(5)
            print(f"Temporal Cluster {c} top genres:")
            print(top5.to_string(), "\n")

        # Plot centroid bar chart
        plt.figure(figsize=(10, 6))
        year_centroids.T.plot.bar(ax=plt.gca())
        plt.ylabel("Avg. Proportion")
        plt.title("Genre Profiles of Temporal Clusters")
        plt.tight_layout()
        plt.savefig("results/temporal_cluster_profiles.png")
        plt.close()

        if 'country' in df.columns:
            tmp2 = df[['country', 'listed_in']].dropna()
            tmp2['country_list'] = tmp2['country'].str.split(',\s*')
            tmp2['genre_list'] = tmp2['listed_in'].str.split(',\s*')
            tmp2 = tmp2.explode('country_list').explode('genre_list')
            country_genre = tmp2.groupby(['country_list', 'genre_list']).size().unstack(fill_value=0)
            country_genre_norm = country_genre.div(country_genre.sum(axis=1), axis=0)

            # choose best k for countries
            best_k_countries = self._choose_k(country_genre_norm, k_min=2, k_max=15, tag="countries")
            best_k_countries = 5

            pca2 = PCA(n_components=2, random_state=0)
            country_coords = pca2.fit_transform(country_genre_norm)
            kmeans_c = KMeans(n_clusters=best_k_countries, random_state=0).fit(country_coords)
            country_labels = kmeans_c.labels_

            plt.figure(figsize=(12, 10))
            colors2 = distinctipy.get_colors(best_k_countries)
            for c in range(best_k_countries):
                mask = (country_labels == c)
                plt.scatter(country_coords[mask, 0], country_coords[mask, 1],
                            color=colors2[c], label=f'Cluster {c}')

            texts = []
            for idx, country in enumerate(country_genre_norm.index):
                x, y = country_coords[idx]
                texts.append(
                    plt.text(x, y, country, fontsize=7, alpha=0.8)
                )

            adjust_text(
                texts,
                only_move={'points': 'xy', 'texts': 'xy'},
                force_text=(0.5, 0.5),
                force_points=(0.5, 0.5),
                expand_text=(1.2, 1.2),
                expand_points=(1.2, 1.2),
                arrowprops=dict(arrowstyle='-', color='black', lw=0.5, alpha=0.5)
            )

            plt.title("Regional Clusters of Countries by Genre Distribution")
            plt.legend()
            plt.tight_layout()
            plt.savefig("results/sub3_regional_clusters.png")
            plt.close()

            pd.DataFrame({
                'country': country_genre_norm.index,
                'cluster': country_labels
            }).to_excel("results/regional_clusters.xlsx", index=False)

            # ANOVA for countries
            print("\n=== ANOVA: Regional clusters ===")
            for genre in country_genre_norm.columns:
                groups = [country_genre_norm[country_labels==c][genre] for c in np.unique(country_labels)]
                stat,p = f_oneway(*groups)
                if p < 0.01:
                    print(f"{genre:20s} p={p:.2e}")


            # Profile regional clusters
            country_centroids = (
                country_genre_norm
                .assign(cluster=country_labels)
                .groupby('cluster')
                .mean()
            )
            for c, row in country_centroids.iterrows():
                top5 = row.sort_values(ascending=False).head(5)
                print(f"Regional Cluster {c} top genres:")
                print(top5.to_string(), "\n")

            # Plot centroid bar chart
            plt.figure(figsize=(12, 6))
            country_centroids.T.plot.bar(ax=plt.gca())
            plt.ylabel("Avg. Proportion")
            plt.title("Genre Profiles of Regional Clusters")
            plt.tight_layout()
            plt.savefig("results/regional_cluster_profiles.png")
            plt.close()


        else:
            logger.warning("No country column—skipping regional analysis")

        logger.info("sub3 analysis complete")


    @staticmethod
    def _choose_k(X, k_min=2, k_max=10, tag=""):
        """
        Runs KMeans for k in [k_min..k_max], saves:
         - elbow plot (inertia vs k) to results/elbow_{tag}.png
         - silhouette plot to results/silhouette_{tag}.png
        Returns the k with the highest silhouette score.
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        inertias, silhouettes = [], []
        Ks = list(range(k_min, k_max + 1))
        for k in Ks:
            km = KMeans(n_clusters=k, random_state=0).fit(X)
            inertias.append(km.inertia_)
            silhouettes.append(silhouette_score(X, km.labels_))

        # elbow
        plt.figure(figsize=(6,4))
        plt.plot(Ks, inertias, marker='o')
        plt.title(f"Elbow Plot ({tag})")
        plt.xlabel("k")
        plt.ylabel("Inertia")
        plt.tight_layout()
        plt.savefig(f"results/elbow_{tag}.png")
        plt.close()

        # silhouette
        plt.figure(figsize=(6,4))
        plt.plot(Ks, silhouettes, marker='o')
        plt.title(f"Silhouette Scores ({tag})")
        plt.xlabel("k")
        plt.ylabel("Silhouette Score")
        plt.tight_layout()
        plt.savefig(f"results/silhouette_{tag}.png")
        plt.close()

        # pick best by silhouette
        best_k = Ks[silhouettes.index(max(silhouettes))]
        logger.info(f"Best k for {tag}: {best_k}")
        return best_k
