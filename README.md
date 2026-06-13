# Star-Galaxy-Classification-Using-k-Nearest-Neighbour
## Objective
The core objective of this project is to build an automated machine learning classifier that can accurately categorize astronomical objects into one of three distinct classes: Stars, Galaxies, or Quasars (QSOs).
Data Source & Domains

The data is sourced from the **Sloan Digital Sky Survey** (SDSS - Skyserver), a major multi-spectral imaging and spectroscopic redshift survey. This is a classic multi-class classification problem intersecting data science and astrophysics (astroinformatics).

## Problem Nuances
- Astrometry vs. Photometry: The dataset contains spatial coordinates (`ra`, `dec`) alongside brightness measurements across five different filters (`u`, `g`, `r`, `i`, `z`) known as the SDSS photometric system.

- Class Imbalance: Galaxies (4,998) and Stars (4,152) dominate the dataset, while Quasars are significantly rarer (850). This class imbalance requires a strategy like stratified splitting to ensure the model learns robust boundaries for the minority class.

## Data Auditing and Cleanup
An initial verification check shows the shape of the dataset is (10000,18). The script explicitly scans for null values using `df.isnull().sum()` and clean drops any rows containing missing entries with `df.dropna()`, confirming a post-cleanup shape of (10000,18)—meaning the source dataset was already structurally complete.

## Domain-Driven Feature Engineering & Selection
The script splits the 18 raw columns into Target Variables, Informative Predictive Features, and Metadata Artifacts:
- Target: `class` (Encoded via LabelEncoder to mapping: `{'GALAXY': 0, 'QSO': 1, 'STAR': 2}`).
- Predictive Features: * `ra`, `dec`: Right Ascension and Declination (celestial coordinates).
  - `u`, `g`, `r`, `i`, `z`: Magnitudes corresponding to Ultraviolet, Green, Red, Near-Infrared, and Infrared filters.
  - redshift: The spectroscopic measurement of how much the wavelength of light from an object has stretched, indicating distance and velocity.
- Excluded Metadata: Identifiers like `objid`, `specobjid`, `run`, `rerun`, `camcol`, `field`, `plate`, `mjd`, and `fiberid` are deliberately filtered out. These variables represent camera configurations, data release batches, or processing dates, which carry zero physical meaning regarding the identity of the celestial object. Including them would risk introducing data leakage or overfit noise.

## Data Splitting
The data is split into a 70% training set (7,000 samples) and a 30% test set (3,000 samples). Crucially, the argument `stratify=y` is utilized, preserving the baseline proportion of Galaxies, Stars, and Quasars across both splits to guard against evaluation bias.

## Feature Scaling
Because kNN calculates distances (e.g., Euclidean or Manhattan) between data points to infer similarity, it is highly sensitive to feature scales. A variable with a naturally wide numeric range would heavily dominate the distance metric over a narrow scale variable. The pipeline solves this by implementing `StandardScaler`, centering the features to a mean of 0 and a standard deviation of 1.

## Hyperparameter Tuning (Grid Search)
Instead of relying on arbitrary default variables, kNN parameters are exhaustively optimized over a multidimensional search space using `GridSearchCV` with 5-fold cross-validation:
- `n_neighbors`: Checked integers from 1 to 15.
- `weights`: Evaluated `uniform` (all neighbors vote equally) vs. `distance` (closer neighbors exert more influence).
- `metric`: Tested distance equations (`euclidean`, `manhattan`, `minkowski`).

The Winning Combination discovered:
- metric: `manhattan`
- n_neighbors: 4
- weights: `distance`
- Best Cross-Validation Score: **92.07%**

## Classification Report Analysis
- **Quasars** (QSO): Attained an extraordinary precision of 0.99. If the model declares an object is a Quasar, it is correct 99% of the time.
- **Stars**: Achieved a high recall (0.96), meaning the model successfully captures nearly all actual stars in the dataset, though at a slight cost to precision (0.89) due to minor confusion with galaxies.

## Feature Importance via Permutation
Because kNN lacks native coefficients or feature weights, Permutation Importance was applied on the test set. This technique shuffles a single feature column and measures how much the model accuracy drops.
- The Astrophysics Validation: `redshift` is overwhelmingly the most vital feature, commanding an importance score of `0.1646`. This aligns perfectly with cosmological physics: stars within our own galaxy have negligible or zero redshift, galaxies have moderate redshift based on distance, and quasars—being highly distant, energetic active galactic nuclei—exhibit massive redshift values.
- The ultraviolet (`u`) and far-infrared (`z`) bands follow in importance, capturing the extreme edges of the light spectrum that differentiate highly energetic active objects from older stellar populations. Spatial coordinates (`ra`, `dec`) hover near zero importance, logically proving that where we look in the sky has little bearing on what physical object is being observed.
