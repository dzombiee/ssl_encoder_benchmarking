# Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### Step 1: Setup Environment (2 min)

```bash
# Clone or navigate to the project
cd /Users/rishabhsharma/Documents/Research_New

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Verify Setup (1 min)

```bash
# Run system test
python test_setup.py
```

You should see:
```
✓ All tests passed! System is ready.
```

### Step 3: Create Data Splits (1 min)

```bash
# Create cold-start splits
python src/data/preprocessing.py
```

This creates:
- `data/processed/train_interactions.jsonl` - Training interactions (warm items only)
- `data/processed/val_interactions.jsonl` - Validation interactions
- `data/processed/test_interactions.jsonl` - Test interactions (warm + cold)
- `data/processed/cold_items.txt` - List of cold items (15%)
- `data/processed/warm_items.txt` - List of warm items (85%)
- `data/processed/item_metadata.jsonl` - All item metadata

### Step 4: Train Your First Model (5-30 min depending on GPU)

```bash
# Train SimCSE (fastest SSL model)
python src/train_ssl.py \
    --model_type simcse \
    --config configs/simcse_config.yaml \
    --output_dir experiments/simcse
```

This will:
1. Load item metadata
2. Train SimCSE encoder for 5 epochs
3. Extract embeddings for all items
4. Save model checkpoint and embeddings

### Step 5: Evaluate (2 min)

```bash
# Evaluate on cold-start items
python src/evaluate_coldstart.py \
    --embeddings experiments/simcse/item_embeddings.npz \
    --model_name simcse \
    --output_dir experiments/results
```

You should see metrics like:
```
NDCG:
  ndcg@5  : 0.0856 ± 0.0023
  ndcg@10 : 0.0912 ± 0.0019

Recall:
  recall@5 : 0.0734 ± 0.0031
  recall@10: 0.1245 ± 0.0028
```

## 🎯 What's Next?

### Train All Models (Full Pipeline)

```bash
# This will train and evaluate all 7 models (5 SSL + 2 baselines)
# Takes 2-4 hours depending on GPU
bash run_pipeline.sh
```

### Compare Results

After running the full pipeline:

```bash
python src/compare_results.py \
    --results_dir experiments/results \
    --output_dir experiments/results/comparison
```

This generates:
- Comparison tables (CSV)
- Visualization plots (PNG)
- Statistical analysis

## 📊 Understanding Results

### Cold Items (Primary Metric)
These are items with **zero training interactions**. This is the main evaluation:
- **NDCG@10**: 0.05-0.15 (good SSL models should be in this range)
- **Recall@10**: 0.10-0.20

### Warm Items (Secondary Metric)
These are items with training interactions. Used to verify embeddings work well in general:
- **NDCG@10**: 0.30-0.50 (should be much higher than cold)

### What to Expect
- **Random baseline**: NDCG@10 ~0.01-0.02
- **TF-IDF baseline**: NDCG@10 ~0.03-0.05
- **SSL models**: NDCG@10 ~0.05-0.15 (3-5x better than random)

## 🔧 Troubleshooting

### "Out of memory" error
```yaml
# Edit configs/base_config.yaml
training:
  batch_size: 64  # Reduce from 128
  
model:
  backbone: "distilbert-base-uncased"  # Use smaller model
```

### Training is too slow
```yaml
# Edit model config (e.g., configs/simcse_config.yaml)
training:
  num_epochs: 3  # Reduce from 5-10

preprocessing:
  max_text_length: 128  # Reduce from 256
```

### Models not improving
1. Check your data: `head -5 data/processed/item_metadata.jsonl`
   - Make sure items have good text (title + description)
2. Increase epochs: Set `num_epochs: 10` in config
3. Try different temperature: `temperature: 0.1` (for contrastive models)

## 💡 Pro Tips

### 1. Start Small
Train just SimCSE first to validate the pipeline:
```bash
python src/train_ssl.py --model_type simcse --config configs/simcse_config.yaml --output_dir experiments/simcse
```

### 2. Monitor Training
Watch the loss decreasing:
- SimCSE/SimCLR: Loss should go from ~6.0 to ~3.0
- TSDAE: Loss should go from ~8.0 to ~4.0
- MLM: Loss should go from ~9.0 to ~5.0

### 3. Quick Experiments
For rapid prototyping, reduce:
- `cold_item_ratio: 0.10` (fewer cold items = faster eval)
- `num_negatives: 499` (500 total candidates = faster ranking)
- `num_epochs: 3` (faster training)

### 4. Best Practices
- Always run baselines (Random + TF-IDF) first
- Run multiple seeds (3-5) for statistical significance
- Save embeddings for later analysis
- Keep detailed logs of hyperparameters

## 📁 File Organization

After running the pipeline, you'll have:

```
experiments/
├── random/
│   └── item_embeddings.npz
├── tfidf/
│   └── item_embeddings.npz
├── simcse/
│   ├── config.json
│   ├── best_model.pt
│   └── item_embeddings.npz
├── simclr/
│   └── ...
├── tsdae/
│   └── ...
├── mlm/
│   └── ...
├── multiview/
│   └── ...
└── results/
    ├── random_results.json
    ├── tfidf_results.json
    ├── simcse_results.json
    ├── ...
    └── comparison/
        ├── cold_items_comparison.png
        ├── warm_items_comparison.png
        └── cold_items_ndcg10.csv
```

## 🎓 For Your Thesis

### Key Results to Report

1. **Main Table**: NDCG@10 for all models on cold items
2. **Comparison Plot**: Bar chart comparing all models
3. **Cold vs Warm**: Show performance gap
4. **Ablations**: Different pooling, temperatures, etc.

### Statistical Significance

Run with multiple seeds:
```bash
for seed in 42 123 456; do
    # Update seed in config
    # Train model
    # Evaluate
done
```

Then use paired t-test to compare models.

## 🆘 Need Help?

1. Check the main [README.md](README.md)
2. Look at error logs in console output
3. Verify data format: `python test_setup.py`
4. Check configs match your data

---

**Happy experimenting! 🚀**
