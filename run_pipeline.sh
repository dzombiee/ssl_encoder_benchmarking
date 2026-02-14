#!/bin/bash

# Master script to run the entire SSL cold-start recommendation pipeline
# Usage: bash run_pipeline.sh

set -e  # Exit on error

echo "======================================================================"
echo "SSL Cold-Start Recommendation Pipeline"
echo "======================================================================"

# Configuration
SEED=42
COLD_RATIO=0.15
DATA_DIR="data/processed"
EXPERIMENTS_DIR="experiments"

# Step 1: Create cold-start splits
echo ""
echo "Step 1: Creating cold-start data splits..."
echo "----------------------------------------------------------------------"
python src/data/preprocessing.py

# Step 2: Train baselines
echo ""
echo "Step 2: Training baseline models..."
echo "----------------------------------------------------------------------"

echo "  Training Random baseline..."
python src/train_baselines.py \
    --baseline_type random \
    --metadata_path ${DATA_DIR}/item_metadata.jsonl \
    --output_dir ${EXPERIMENTS_DIR}/random

echo "  Training Vanilla BERT baseline..."
python src/train_baselines.py \
    --baseline_type vanilla_bert \
    --metadata_path ${DATA_DIR}/item_metadata.jsonl \
    --output_dir ${EXPERIMENTS_DIR}/vanilla_bert

echo "  Training Sentence-BERT baseline..."
python src/train_baselines.py \
    --baseline_type sbert \
    --metadata_path ${DATA_DIR}/item_metadata.jsonl \
    --output_dir ${EXPERIMENTS_DIR}/sbert

# Step 3: Train SSL models
echo ""
echo "Step 3: Training SSL models..."
echo "----------------------------------------------------------------------"

# SimCSE (dropout)
echo "  Training SimCSE..."
python src/train_ssl.py \
    --model_type simcse \
    --config configs/simcse_config.yaml \
    --output_dir ${EXPERIMENTS_DIR}/simcse

# SimCLR
echo "  Training SimCLR..."
python src/train_ssl.py \
    --model_type simclr \
    --config configs/simclr_config.yaml \
    --output_dir ${EXPERIMENTS_DIR}/simclr

# TSDAE
echo "  Training TSDAE..."
python src/train_ssl.py \
    --model_type tsdae \
    --config configs/tsdae_config.yaml \
    --output_dir ${EXPERIMENTS_DIR}/tsdae

# MLM
echo "  Training MLM..."
python src/train_ssl.py \
    --model_type mlm \
    --config configs/mlm_config.yaml \
    --output_dir ${EXPERIMENTS_DIR}/mlm

# Multi-view
echo "  Training Multi-view Contrastive..."
python src/train_ssl.py \
    --model_type multiview \
    --config configs/multiview_config.yaml \
    --output_dir ${EXPERIMENTS_DIR}/multiview

# Step 4: Evaluate all models
echo ""
echo "Step 4: Evaluating all models..."
echo "----------------------------------------------------------------------"

RESULTS_DIR="${EXPERIMENTS_DIR}/results"
mkdir -p ${RESULTS_DIR}

for MODEL in random tfidf sbert simcse simclr tsdae mlm multiview; do
    echo "  Evaluating ${MODEL}..."
    python src/evaluate_coldstart.py \
        --embeddings ${EXPERIMENTS_DIR}/${MODEL}/item_embeddings.npz \
        --model_name ${MODEL} \
        --output_dir ${RESULTS_DIR}
done

# Step 5: Compare results
echo ""
echo "Step 5: Comparing all models..."
echo "----------------------------------------------------------------------"
python src/compare_results.py \
    --results_dir ${RESULTS_DIR} \
    --output_dir ${RESULTS_DIR}/comparison

echo ""
echo "======================================================================"
echo "Pipeline completed successfully!"
echo "======================================================================"
echo "Results saved to: ${RESULTS_DIR}"
