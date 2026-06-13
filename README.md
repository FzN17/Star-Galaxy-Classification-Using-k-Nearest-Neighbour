# Star-Galaxy-Classification-Using-k-Nearest-Neighbour
## Objective
The core objective of this project is to build an automated machine learning classifier that can accurately categorize astronomical objects into one of three distinct classes: Stars, Galaxies, or Quasars (QSOs).
Data Source & Domains

The data is sourced from the Sloan Digital Sky Survey (SDSS - Skyserver), a major multi-spectral imaging and spectroscopic redshift survey. This is a classic multi-class classification problem intersecting data science and astrophysics (astroinformatics).

## Problem Nuances
- Astrometry vs. Photometry: The dataset contains spatial coordinates (ra, dec) alongside brightness measurements across five different filters (u, g, r, i, z) known as the SDSS photometric system.

- Class Imbalance: Galaxies (4,998) and Stars (4,152) dominate the dataset, while Quasars are significantly rarer (850). This class imbalance requires a strategy like stratified splitting to ensure the model learns robust boundaries for the minority class.
