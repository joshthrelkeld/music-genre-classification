import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

data_path = Path(__file__).resolve().parent.parent / "data" / "data.csv"
df = pd.read_csv(data_path)

genre_stats = df.groupby("label").agg(
    Mean_Centroid=("spectral_centroid", "mean"),
    STDEV_Centroid=("spectral_centroid", "std"),
    Mean_Bandwidth=("spectral_bandwidth", "mean")
).reset_index()

genre_stats["label"] = genre_stats["label"].replace("hiphop", "Hip Hop")
genre_stats["label"] = genre_stats["label"].apply(
    lambda x: x if x == "Hip Hop" else x.capitalize()
)
genre_stats = genre_stats.rename(columns={"label": "Genre"})

colors = {
    'Blues': '#1f77b4',
    'Classical': '#9467bd',
    'Country': '#8c564b',
    'Disco': '#e377c2',
    'Hip Hop': '#ff7f0e',
    'Jazz': '#2ca02c',
    'Metal': '#d62728',
    'Pop': '#f7b731',
    'Reggae': '#17becf',
    'Rock': '#7f7f7f'
}

# Plot each genre in centroid-bandwidth space to assess whether genres cluster
# into distinct regions or show substantial overlap
fig, ax = plt.subplots(figsize=(10, 7))

for _, row in genre_stats.iterrows():
    genre = row['Genre']
    ax.scatter(
        row['Mean_Centroid'],
        row['Mean_Bandwidth'],
        color=colors[genre],
        s=120,
        edgecolors='white',
        linewidths=0.8,
        zorder=3
    )
    ax.annotate(
        genre,
        xy=(row['Mean_Centroid'], row['Mean_Bandwidth']),
        xytext=(6, 4),
        textcoords='offset points',
        fontsize=9,
        color=colors[genre],
        fontweight='bold'
    )

# Show within-genre spread in centroid values along the x-axis
ax.errorbar(
    genre_stats['Mean_Centroid'],
    genre_stats['Mean_Bandwidth'],
    xerr=genre_stats['STDEV_Centroid'],
    fmt='none',
    ecolor='gray',
    alpha=0.4,
    capsize=3,
    zorder=2
)

ax.set_xlabel('Mean Spectral Centroid (Hz)', fontsize=12)
ax.set_ylabel('Mean Spectral Bandwidth (Hz)', fontsize=12)
ax.set_title('Spectral Centroid vs. Bandwidth by Genre', fontsize=14, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.4)
ax.set_facecolor('#f9f9f9')

plt.tight_layout()
plt.savefig('../figures/centroid_vs_bandwidth.png', dpi=300, bbox_inches='tight')
plt.show()