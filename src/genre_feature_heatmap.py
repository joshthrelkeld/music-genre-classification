"""Generates the Genre Feature Profile Heatmap (z-score normalized)."""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

data_path = Path(__file__).resolve().parent.parent / "data" / "data.csv"
df = pd.read_csv(data_path)

# Use a small, interpretable feature set to compare how genres differ in both
# cepstral structure (MFCCs) and broader spectral characteristics
features = [
    "mfcc1", "mfcc2",                          # MFCC-1 and MFCC-2 are visually emphasized in the plot below
    "spectral_centroid", "spectral_bandwidth", "chroma_stft"
]

feature_labels = {
    "mfcc1":              "MFCC-1",
    "mfcc2":              "MFCC-2",
    "spectral_centroid":  "Spectral Centroid",
    "spectral_bandwidth": "Spectral Bandwidth",
    "chroma_stft":        "Chroma (STFT)"
}

# Average each feature within genre, then z-score across genres so features with
# different raw scales can be compared on the same heatmap
genre_means = df.groupby("label")[features].mean()
genre_means_z = (genre_means - genre_means.mean()) / genre_means.std()
genre_means_z = genre_means_z.rename(columns=feature_labels)

# Capitalize genre labels for readability
genre_means_z.index = genre_means_z.index.str.capitalize()

# Look for distinct feature profiles across genres rather than uniform values
# Clear differences would support the idea that these features contain meaningful
# structure for genre classification
fig, ax = plt.subplots(figsize=(10, 7))

sns.heatmap(
    genre_means_z,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    linecolor="white",
    cbar_kws={"label": "Z-Score", "shrink": 0.8},
    ax=ax
)

# Visually emphasize MFCC-1 and MFCC-2 because they are central to the
# interpretation of cross-genre differences in this analysis
# col_idx 0 and 1 correspond to MFCC-1 and MFCC-2, which must remain first in the features list above
for col_idx in [0, 1]:
    ax.add_patch(mpatches.FancyBboxPatch(
        (col_idx, 0),
        1, len(genre_means_z),
        boxstyle="square,pad=0",
        linewidth=2.5,
        edgecolor="#222222",
        facecolor="none",
        transform=ax.transData,
        clip_on=False
    ))

ax.set_title(
    "Genre Feature Profile Heatmap (Z-Score Normalized)",
    fontsize=14,
    fontweight="bold",
    pad=14
)
ax.set_xlabel("Feature", fontsize=11, labelpad=10)
ax.set_ylabel("Genre", fontsize=11, labelpad=10)
ax.tick_params(axis="x", labelsize=10, rotation=20)
ax.tick_params(axis="y", labelsize=10, rotation=0)

plt.tight_layout()
fig_path = Path(__file__).resolve().parent.parent / "figures" / "genre_heatmap.png"
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.show()