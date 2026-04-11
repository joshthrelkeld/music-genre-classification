import pandas as pd
import matplotlib.pyplot as plt

# This scatterplot will be used to compare genres using spectral centroid and bandwidth
# The goal is to see whether simple spectral features separate genres in a way
# that helps explain classification performance

# Centroid reflects perceived brightness, while bandwidth captures how broadly
# energy is distributed across frequencies
# Precomputed genre-level summary statistics used for a cleaner standalone plot
data = {
    'Genre': [
        'Blues', 'Classical', 'Country', 'Disco', 'Hip Hop',
        'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock'
    ],
    'Mean_Centroid': [
        1727.655, 1353.991, 1896.096, 2619.974, 2524.614,
        1792.404, 2602.175, 3073.664, 2185.111, 2242.657
    ],
    'STDEV_Centroid': [
        515.546, 348.305, 575.795, 478.742, 479.309,
        680.607, 368.584, 582.062, 626.599, 483.838
    ],
    'Mean_Bandwidth': [
        1931, 1522, 2099, 2513, 2514,
        2021, 2242, 3008, 2311, 2263
    ]
}

df = pd.DataFrame(data)

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
# If genres fall into distinct parts of the plot, centroid and bandwidth may
# carry useful signal for classification. Strong overlap would suggest these
# features are informative but not sufficient on their own
fig, ax = plt.subplots(figsize=(10, 7))

for _, row in df.iterrows():
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

    # Label points directly so genre positions can be compared without a legend
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
# Larger spreads suggest a genre is less tightly clustered and may be harder to separate
ax.errorbar(
    df['Mean_Centroid'],
    df['Mean_Bandwidth'],
    xerr=df['STDEV_Centroid'],
    fmt='none',
    ecolor='gray',
    alpha=0.4,
    capsize=3,
    zorder=2
)

# Isolated points within the figure suggest stronger genre-specific structure, while
# overlapping points suggest acoustic similarity in these two dimensions

ax.set_xlabel('Mean Spectral Centroid (Hz)', fontsize=12)
ax.set_ylabel('Mean Spectral Bandwidth (Hz)', fontsize=12)
ax.set_title('Spectral Centroid vs. Bandwidth by Genre', fontsize=14, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.4)
ax.set_facecolor('#f9f9f9')

plt.tight_layout()
plt.savefig('centroid_vs_bandwidth.png', dpi=300, bbox_inches='tight')
plt.show()