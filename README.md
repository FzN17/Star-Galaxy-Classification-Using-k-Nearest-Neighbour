# Star-Galaxy-Classification-Using-k-Nearest-Neighbour
## Objective
The core objective of this project is to build an automated machine learning classifier that can accurately categorize astronomical objects into one of three distinct classes: Stars, Galaxies, or Quasars (QSOs).
Data Source & Domains

The data is sourced from the **Sloan Digital Sky Survey** (SDSS - Skyserver), a major multi-spectral imaging and spectroscopic redshift survey. This is a classic multi-class classification problem intersecting data science and astrophysics (astroinformatics).

## Problem Nuances
- Astrometry vs. Photometry: The dataset contains spatial coordinates (ra, dec) alongside brightness measurements across five different filters (u, g, r, i, z) known as the SDSS photometric system.

- Class Imbalance: Galaxies (4,998) and Stars (4,152) dominate the dataset, while Quasars are significantly rarer (850). This class imbalance requires a strategy like stratified splitting to ensure the model learns robust boundaries for the minority class.

## Data Auditing and Cleanup
An initial verification check shows the shape of the dataset is (10000,18). The script explicitly scans for null values using `df.isnull().sum()` and clean drops any rows containing missing entries with `df.dropna()`, confirming a post-cleanup shape of (10000,18)—meaning the source dataset was already structurally complete.

## Domain-Driven Feature Engineering & Selection
The script splits the 18 raw columns into Target Variables, Informative Predictive Features, and Metadata Artifacts:
- Target: class (Encoded via LabelEncoder to mapping: {'GALAXY': 0, 'QSO': 1, 'STAR': 2}).
- Predictive Features: * ra, dec: Right Ascension and Declination (celestial coordinates).
 - u, g, r, i, z: Magnitudes corresponding to Ultraviolet, Green, Red, Near-Infrared, and Infrared filters.
 - redshift: The spectroscopic measurement of how much the wavelength of light from an object has stretched, indicating distance and velocity.
- Excluded Metadata: Identifiers like objid, specobjid, run, rerun, camcol, field, plate, mjd, and fiberid are deliberately filtered out. These variables represent camera configurations, data release batches, or processing dates, which carry zero physical meaning regarding the identity of the celestial object. Including them would risk introducing data leakage or overfit noise.
