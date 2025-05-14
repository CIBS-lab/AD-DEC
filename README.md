# AD-DEC

**AD-DEC** is a TensorFlow-based implementation of Deep Embedded Clustering (DEC) applied to high-risk individuals from the Wisconsin Registry for Alzheimer's Prevention (WRAP) study. This repository enables researchers to identify biologically meaningful Alzheimer’s Disease (AD) risk phenotypes using statistical summaries of blood biomarkers, cognitive scores, and clinical features.

## 🔍 Overview

This project applies DEC to a curated dataset of WRAP participants enriched for AD risk. Latent clusters are identified and validated with demographic, clinical, and molecular profiles — including GFAP, glucose, insulin, and LDL levels. 

The codebase is adapted from the GitHub repositories of **de Kok** and **Castela Forte**, with modifications tailored for longitudinal Alzheimer’s data and DEC optimization in TensorFlow.

## 📁 Repository Structure

- `scripts/`: Preprocessing, DEC model architecture, training loop, and evaluation.
- `notebooks/`: Interactive Jupyter notebook for end-to-end DEC pipeline walkthrough.
- `data/`: (Optional) Mock input data resembling WRAP data structure.

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/AD-DEC.git
   cd AD-DEC
