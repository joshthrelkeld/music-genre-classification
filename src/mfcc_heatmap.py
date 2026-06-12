import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('../data/data.csv')

# MFCCs capture the timbre of audio, which is strongly tied to genre
mfcc_cols = [f'mfcc{i}' for i in range(1, 21)]

# For each genre, aggregate MFCCs to reveal spectral patterns per class
mfcc_means = df.groupby('label')[mfcc_cols].mean()

# Reorder genres for consistent comparison (avoiding arbitrary pandas ordering)
genre_order = ['blues', 'classical', 'country', 'disco', 'hiphop',
               'jazz', 'metal', 'pop', 'reggae', 'rock']
mfcc_means = mfcc_means.loc[genre_order]

mfcc_means.index = ['Blues', 'Classical', 'Country', 'Disco', 'Hip Hop',
                    'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']

# Create visual representation of cross-genre differences in MFCC profiles to assess whether
# spectral features provide separable classification structure
fig, ax = plt.subplots(figsize=(14, 7))

im = ax.imshow(
    mfcc_means.values,
    aspect='auto',
    cmap='coolwarm',      # shows how values differ across genres
    interpolation='nearest'
)

# Hypothesis: if genres are acoustically distinct, their MFCC profiles should show
# consistent differences across coefficients (e.g., certain coefficients elevated),
# rather than appearing uniform across rows
# If MFCC values appear similar across genres (homogeneous color distribution),
# this suggests weak separability and may explain poor model performance

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Mean MFCC Coefficient Value', fontsize=11)

ax.set_xticks(np.arange(len(mfcc_cols)))
ax.set_xticklabels([f'MFCC-{i}' for i in range(1, 21)], rotation=45, ha='right', fontsize=9)

ax.set_yticks(np.arange(len(mfcc_means.index)))
ax.set_yticklabels(mfcc_means.index, fontsize=10)

ax.set_xlabel('MFCC Coefficient', fontsize=12)
ax.set_ylabel('Genre', fontsize=12)
ax.set_title('Mean MFCC Coefficients by Genre', fontsize=14, fontweight='bold')

ax.set_xticks(np.arange(-0.5, 20, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
ax.grid(which='minor', color='white', linewidth=1.5)
ax.tick_params(which='minor', bottom=False, left=False)

plt.tight_layout()
plt.savefig('../figures/mfcc_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()