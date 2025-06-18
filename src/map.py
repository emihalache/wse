import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import os

# after loading
clusters = pd.read_excel("../results/regional_clusters.xlsx")
clusters["country_clean"] = clusters["country"].str.strip()

world = gpd.read_file("https://datahub.io/core/geo-countries/r/countries.geojson")

# build the two sets
world_names = set(world["name"])
cluster_names = set(clusters["country_clean"])

# compute differences
only_in_world = sorted(world_names - cluster_names)
only_in_clusters = sorted(cluster_names - world_names)

# print them
print("Countries in GeoJSON but NOT in our clusters:")
for c in only_in_world:
    print("  ", c)
print("\nCountries in our clusters but NOT in GeoJSON:")
for c in only_in_clusters:
    print("  ", c)

name_map = {
    "United States": "United States of America",
    "Czech Republic": "Czechia",
    "Serbia": "Republic of Serbia",
    "Soviet Union": "Russia",
    "Hong Kong": "Hong Kong S.A.R.",
    "West Germany": "Germany"
}

clusters["country_clean"] = clusters["country_clean"].replace(name_map)

map_df = world.merge(
    clusters,
    how="left",
    left_on="name",
    right_on="country_clean"
)

fig, ax = plt.subplots(1, 1, figsize=(14, 8))
map_df.plot(
    column="cluster",
    categorical=True,
    legend=True,
    legend_kwds={"title": "Cluster"},
    missing_kwds={
        "color": "#f0f0f0",
        "label": "No data"
    },
    ax=ax
)
ax.set_axis_off()
ax.set_title("Netflix Genre-Profile Clusters by Country")
plt.tight_layout()
plt.savefig("../results/country_cluster_map.png")
