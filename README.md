# Benchmarking SSL Models for Cold-Start Item Recommendation

This repository contains the complete implementation for the thesis: **"Benchmarking SSL Models for Cold-Start Item Recommendation using Product Metadata"**.

## 📋 Overview

This project benchmarks **five Self-Supervised Learning (SSL)** item encoders trained on product metadata for cold-start recommendation. The goal is to evaluate which SSL pretraining method produces the best item embeddings for recommending items with zero training interactions.

### SSL Models Implemented

1. **SimCSE (Dropout)**: Contrastive learning with dropout as augmentation
2. **SimCLR**: Contrastive learning with text augmentations (token dropout, crop, shuffle)
3. **TSDAE**: Transformer-based Sequential Denoising AutoEncoder
4. **MLM**: Masked Language Modeling (BERT fine-tuning)
5. **Multi-View Contrastive**: Multi-view learning across title, description, and attributes

### Baselines

- **Random Embeddings**: Random normalized vectors
- **TF-IDF**: Classic content-based approach

## 🏗️ Project Structure

```
Research_New/
├── configs/                     # Configuration files
│   ├── base_config.yaml        # Base configuration
│   ├── simcse_config.yaml      # SimCSE specific config
│   ├── simclr_config.yaml      # SimCLR specific config
│   ├── tsdae_config.yaml       # TSDAE specific config
│   ├── mlm_config.yaml         # MLM specific config
│   └── multiview_config.yaml   # Multi-view specific config
│
├── data/                        # Raw data
│   ├── All_Beauty.jsonl        # User-item interactions
│   └── meta_All_Beauty.jsonl   # Item metadata
│
├── src/                         # Source code
│   ├── data/                    # Data processing
│   │   ├── preprocessing.py    # Cold-start split creation
│   │   ├── datasets.py         # PyTorch datasets
│   │   └── augmentations.py   # Text augmentation functions
│   │
│   ├── models/                  # Model architectures
│   │   ├── base_encoder.py     # Base BERT encoder
│   │   ├── simcse.py          # SimCSE & SimCLR models
│   │   ├── tsdae.py           # TSDAE model
│   │   ├── mlm.py             # MLM model
│   │   ├── multiview.py       # Multi-view contrastive
│   │   ├── baselines.py       # Random & TF-IDF baselines
│   │   └── supervised_scorer.py # MLP scorer for supervised fine-tuning
│   │
│   ├── evaluation/              # Evaluation framework
│   │   ├── metrics.py          # Recall@K, NDCG@K, etc.
│   │   └── zero_shot_recommender.py # Dot-product recommender
│   │
│   ├── utils/                   # Utilities
│   │   └── helpers.py          # Helper functions
│   │
│   ├── train_ssl.py            # Train SSL models
│   ├── train_baselines.py      # Train baselines
│   ├── evaluate_coldstart.py   # Evaluate on cold-start items
│   └── compare_results.py      # Compare all models
│
├── experiments/                 # Experiment outputs
│   ├── simcse/                 # SimCSE results
│   ├── simclr/                 # SimCLR results
│   ├── tsdae/                  # TSDAE results
│   ├── mlm/                    # MLM results
│   ├── multiview/              # Multi-view results
│   ├── random/                 # Random baseline
│   ├── tfidf/                  # TF-IDF baseline
│   └── results/                # Evaluation results
│
├── run_pipeline.sh             # Master pipeline script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

Place your Amazon dataset files in the `data/` directory:
- `All_Beauty.jsonl` - User-item interactions
- `meta_All_Beauty.jsonl` - Item metadata

### 3. Run Complete Pipeline

```bash
bash run_pipeline.sh
```

This will:
1. Create cold-start data splits (15% cold items)
2. Train all baseline models
3. Train all 5 SSL models
4. Evaluate all models on cold-start items
5. Generate comparison plots and tables

### 4. Individual Steps (Optional)

If you want to run steps individually:

#### Create Cold-Start Splits
```bash
python src/data/preprocessing.py
```

#### Train Individual SSL Models
```bash
# SimCSE
python src/train_ssl.py \
    --model_type simcse \
    --config configs/simcse_config.yaml \
    --output_dir experiments/simcse

# SimCLR
python src/train_ssl.py \
    --model_type simclr \
    --config configs/simclr_config.yaml \
    --output_dir experiments/simclr

# TSDAE
python src/train_ssl.py \
    --model_type tsdae \
    --config configs/tsdae_config.yaml \
    --output_dir experiments/tsdae

# MLM
python src/train_ssl.py \
    --model_type mlm \
    --config configs/mlm_config.yaml \
    --output_dir experiments/mlm

# Multi-view
python src/train_ssl.py \
    --model_type multiview \
    --config configs/multiview_config.yaml \
    --output_dir experiments/multiview
```

#### Train Baselines
```bash
# Random
python src/train_baselines.py \
    --baseline_type random \
    --metadata_path data/processed/item_metadata.jsonl \
    --output_dir experiments/random

# TF-IDF
python src/train_baselines.py \
    --baseline_type tfidf \
    --metadata_path data/processed/item_metadata.jsonl \
    --output_dir experiments/tfidf
```

#### Evaluate Models
```bash
python src/evaluate_coldstart.py \
    --embeddings experiments/simcse/item_embeddings.npz \
    --model_name simcse \
    --output_dir experiments/results
```

#### Compare Results
```bash
python src/compare_results.py \
    --results_dir experiments/results \
    --output_dir experiments/results/comparison
```

## 📊 Evaluation Protocol

### Cold-Start Split
- **15% of items** are designated as cold items
- All interactions for cold items are **removed from training**
- Metadata for cold items is **retained** for SSL pretraining
- Test users have **≥1 warm item** in their history

### Metrics
- **Recall@K** (K = 5, 10, 20)
- **NDCG@K** (K = 5, 10, 20)
- **Precision@K**
- **Hit Rate@K**
- **MRR** (Mean Reciprocal Rank)

### Zero-Shot Evaluation
- User embedding = mean of item embeddings in user history
- Score(u, i) = dot(user_embedding, item_embedding)
- **No supervised fine-tuning** on interaction data
- Pure representation quality test

### Negative Sampling
- 999 negatives per positive (1000 total candidates)
- Popularity-based sampling (realistic difficulty)

## ⚙️ Configuration

Key parameters in `configs/base_config.yaml`:

```yaml
# Data split
split:
  cold_item_ratio: 0.15
  min_user_interactions: 5
  min_warm_items_per_user: 1

# Model architecture
model:
  backbone: "bert-base-uncased"
  embedding_dim: 256
  pooling_strategy: "mean"

# Training
training:
  batch_size: 128
  learning_rate_backbone: 2.0e-5
  learning_rate_head: 1.0e-3
  num_epochs: 10
  temperature: 0.05  # For contrastive loss

# Evaluation
evaluation:
  k_values: [5, 10, 20]
  num_negatives: 999
  negative_sampling: "popularity"
```

## 📈 Expected Results

### Performance Range (NDCG@10 on Cold Items)
- **Random**: 0.01 - 0.02
- **TF-IDF**: 0.03 - 0.05
- **SSL Models**: 0.05 - 0.15

The SSL models should beat baselines by **3-5x** on cold items while maintaining good performance on warm items (NDCG@10: 0.3-0.5).

## 🔬 Research Questions Addressed

1. **Which SSL method produces the best embeddings for cold-start items?**
2. **How does zero-shot performance compare to baselines?**
3. **What is the performance gap between cold and warm items?**
4. **Are metadata-based embeddings sufficient for cold-start recommendation?**

## 📝 Key Implementation Details

### Text Preprocessing
- Concatenate title + description + attributes
- Lowercase, remove HTML
- Truncate to 256 tokens (BERT max length)
- Filter items with insufficient text

### SSL Pretraining
- All models pretrained on **all items** (warm + cold)
- Only metadata used (no interaction data)
- Same BERT backbone for fair comparison
- Project to 256-dim embeddings

### User Representation
- Mean pooling of item embeddings in user history
- Only warm items used (items with training interactions)
- L2 normalization

## 🛠️ Troubleshooting

### Out of Memory
- Reduce `batch_size` in config files
- Use `distilbert-base-uncased` instead of `bert-base-uncased`
- Reduce `max_text_length` to 128

### Slow Training
- Reduce `num_epochs` to 5
- Use smaller `max_features` for TF-IDF (2000)
- Enable mixed precision training: `use_amp: true`

### Poor Results
- Check data quality (sufficient metadata?)
- Increase training epochs
- Tune temperature for contrastive loss (0.05-0.2)
- Try different pooling strategies

## 📚 Citation

If you use this code for your research, please cite:

```bibtex
@mastersthesis{sharma2026ssl,
  title={Benchmarking SSL Models for Cold-Start Item Recommendation using Product Metadata},
  author={Sharma, Rishabh},
  year={2026},
  school={Your University}
}
```

## 📖 References

1. **SimCSE**: Gao et al., "SimCSE: Simple Contrastive Learning of Sentence Embeddings", EMNLP 2021
2. **SimCLR**: Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations", ICML 2020
3. **TSDAE**: Wang et al., "TSDAE: Using Transformer-based Sequential Denoising Auto-Encoder for Unsupervised Sentence Embedding Learning", ACL 2021
4. **BERT**: Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers", NAACL 2019

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact [your email].

## 📄 License

This project is for academic research purposes.

---

**Note**: This implementation follows best practices for cold-start recommendation research. All design choices are justified in the thesis document.
