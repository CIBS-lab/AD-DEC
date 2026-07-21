# AD-DEC

**AD-DEC** is a TensorFlow-based implementation of Deep Embedded Clustering (DEC) applied to high-risk individuals from the Wisconsin Registry for Alzheimer's Prevention (WRAP) study. This repository enables researchers to identify biologically meaningful Alzheimer’s Disease (AD) risk phenotypes using statistical summaries of blood biomarkers, cognitive scores, and clinical features.

## 🔍 Overview

This project applies DEC to a curated dataset of WRAP participants enriched for AD risk. Latent clusters are identified in WRAP and projected into the diverse Health and Aging Brain Study: Health Disparities (HABS-HD) cohort for external validation, using demographic, clinical, and molecular profiles — including GFAP, glucose, insulin, and LDL levels. 

The codebase is adapted from the GitHub repositories of **de Kok** and **Castela Forte**, with modifications tailored for longitudinal Alzheimer’s data and DEC optimization in TensorFlow.

## 📁 Repository Structure

- `scripts/`: Preprocessing, DEC model architecture, training, internal
  validation and cluster stability, and cardiovascular-risk translation.
- `scripts/coefficients/`: Cited coefficient files for the PREVENT and PCE scores.
- `notebooks/`: (Optional) Interactive Jupyter notebook for the DEC pipeline.
- `data/`: (Optional) Mock input data resembling WRAP data structure.

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/AD-DEC.git
   cd AD-DEC

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Run clustering and cluster stability:
   The DEC model, training, and
   stability computation live in `scripts/clustering_functions.py` (see
   `compute_cluster_stability`, `jaccard_similarity`, `sample_cluster_stability`).

4. Classifier-based validation and SHAP:
   `scripts/validation_functions.py`
   (see `clf_stability_analysis`).

📊 Input Data Format
Expected format is a .csv file where each row corresponds to a single participant, and columns include:

   Biomarker summaries (mean, SD, skew) across timepoints

   Composite cognitive scores

   Demographics: age, sex, APOE genotype

   Optional: MRI features, medication use, family history of AD

Note: Actual WRAP data is not shared in this repository. Synthetic mock input is provided for demonstration.

## 🫀 Cardiovascular risk and phenotype translation

These scripts place the phenotypes on a guideline cardiovascular-risk scale and
support the clinical-translation analyses in the manuscript.

- `scripts/prevent.py`: AHA PREVENT 10-year total CVD risk (Supplementary Tables S12, S13).
- `scripts/pce.py`: 2013 Pooled Cohort Equations, race-based comparison (Supplementary Table S16).
- `scripts/nnt.py`: illustrative NNT by phenotype (Supplementary Table S15); runs out of the box.
- `scripts/coefficients/`: cited coefficient files for PREVENT and PCE.

`nnt.py` runs directly. `prevent.py` and `pce.py` activate once their coefficient
files are populated from the primary sources and verified; see
`scripts/coefficients/README.md`.

## 📏 Cluster stability reporting

Cluster stability is computed in `scripts/clustering_functions.py`
(`compute_cluster_stability`, `jaccard_similarity`, `sample_cluster_stability`),
following de Kok et al. (2024) and Castela Forte et al. (2021). Report the mean
cluster-wise Jaccard across clusters as the whole-model value (0.54 for the
six-phenotype WRAP solution) together with the sample-wise stability (73%). One
phenotype (younger-tau) is highly stable (per-cluster Jaccard near 0.99); this is
not reported as the reproducibility of the whole solution.

📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

🙏 Acknowledgments
Wisconsin Registry for Alzheimer’s Prevention (WRAP), the discovery cohort.

Health and Aging Brain Study: Health Disparities (HABS-HD), the diverse external
validation cohort, a study of the Institute for Translational Research at the
University of North Texas Health Science Center, supported by the National
Institute on Aging. We thank the HABS-HD participants and study team. HABS-HD
data are available from the study team per their access procedures and are not
shared in this repository.

Deep Embedded Clustering by Xie et al.

Adapted from:
de Kok GitHub
Castela Forte GitHub
