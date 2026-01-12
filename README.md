# 🎵 VAE-Based Unsupervised Learning for Hybrid-Language Music Clustering

**Author:** Miftahul Jannah  
**Student ID:** 21241025  

## 📘 Overview
This project implements an **unsupervised clustering pipeline** for multilingual (Bangla-English) music using **Variational Autoencoders (VAE)** and **Conditional VAEs (CVAE)**. The pipeline progressively explores three levels of difficulty:

1. **Easy Task:** Clustering hybrid-language lyrics using TF-IDF features and VAE embeddings.  
2. **Medium Task:** Using **Million Song Dataset (MSD)** audio features and a **Conv-VAE** for MFCC-like timbre and pitch descriptors.  
3. **Hard Task:** A **multi-modal CVAE** combining audio, lyrics, and metadata (genre, language) for comprehensive clustering and analysis.

The study compares **PCA**, **VAE**, and **CVAE** representations under multiple clustering algorithms — *K-Means, Agglomerative, and DBSCAN* — using metrics such as **Silhouette**, **Calinski–Harabasz**, **Davies–Bouldin**, **ARI**, **NMI**, and **Purity**.

---

## 🧠 Key Methods
### 1. Variational Autoencoder (VAE)
A probabilistic model that learns latent representations of music features by optimizing the **Evidence Lower Bound (ELBO)**.  
Used for both lyrics and MSD feature embeddings.

### 2. Convolutional VAE (Conv-VAE)
Designed for **MFCC-like sequences** extracted from the **MSD**.  
Captures local structure across timbre/pitch coefficients using 1D convolutions and transposed convolutions.

### 3. Conditional VAE (CVAE)
Extends VAE by conditioning on side information such as **genre** and **language**, improving disentanglement and cluster alignment.

### 4. Clustering Algorithms
- **K-Means** (baseline)
- **Agglomerative (Ward Linkage)**
- **DBSCAN** (density-based)

---

## 📊 Evaluation Metrics
| Metric | Type | Description |
|---------|------|-------------|
| **Silhouette** | Internal | Measures cohesion and separation between clusters |
| **Calinski–Harabasz** | Internal | Ratio of between-cluster to within-cluster dispersion |
| **Davies–Bouldin** | Internal | Penalizes overlapping clusters |
| **ARI (Adjusted Rand Index)** | External | Measures similarity between clustering and ground-truth labels |
| **NMI (Normalized Mutual Information)** | External | Quantifies mutual dependence between clusters and labels |
| **Purity** | External | Measures label homogeneity within clusters |

---

## 🎧 Datasets Used

### 📝 Hybrid Lyrics Dataset
- **Content:** Bangla and English translations of song lyrics.  
- **Usage:** Easy task for VAE training on text-based features.  
- **Preprocessing:** TF-IDF vectorization → dimensionality reduction via PCA or VAE encoder.

### 🎵 Million Song Dataset (MSD)
- **Source:** [http://millionsongdataset.com/](http://millionsongdataset.com/)  
- **Content:**  
  - Audio analysis features (timbre, pitches, loudness, tempo, duration, etc.)  
  - Metadata (artist, year, genre, song ID)  
- **Note:** MSD does **not** include raw audio; it provides **HDF5-based analysis features** computed from audio (MFCC-like timbre/pitch summaries).  
- **Usage:**  
  - Medium task (Conv-VAE on timbre/pitch features).  
  - Hard task (multimodal fusion with lyrics + metadata).  
- **Feature Extraction Script:** `extract_msd_features.py`

