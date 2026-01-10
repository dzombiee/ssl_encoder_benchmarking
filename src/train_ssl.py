"""
Training script for SSL models.
"""

import argparse
import torch  # type: ignore
from torch.utils.data import DataLoader  # type: ignore
from transformers import AutoTokenizer, get_linear_schedule_with_warmup  # type: ignore
from pathlib import Path
from tqdm import tqdm  # type: ignore

from data.datasets import (
    ItemMetadataDataset,
    ContrastiveDataset,
    MultiViewDataset,
    collate_fn_contrastive,
    collate_fn_multiview,
    collate_fn_tsdae,
    collate_fn_mlm,
)
from data.augmentations import simcse_dropout_augmentation, simclr_augmentation
from models.simcse import SimCSEModel, SimCLRModel
from models.tsdae import TSDAEModel
from models.mlm import MLMModel
from models.multiview import MultiViewContrastiveModel
from utils.helpers import (
    set_seed,
    load_config,
    merge_configs,
    save_json,
    save_embeddings,
    AverageMeter,
    get_device,
)


def train_epoch(model, dataloader, optimizer, scheduler, device, model_type="simcse"):
    """Train for one epoch."""
    model.train()
    loss_meter = AverageMeter()

    use_amp = getattr(dataloader, 'use_amp', False)
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == "cuda" else None

    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast(enabled=True):
                if model_type in ["simcse", "simclr"]:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    num_views = batch["num_views"]
                    loss, _ = model(input_ids, attention_mask, num_views)
                elif model_type == "tsdae":
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    target_ids = batch["target_ids"].to(device)
                    target_attention_mask = batch["target_attention_mask"].to(device)
                    loss, _ = model(input_ids, attention_mask, target_ids, target_attention_mask)
                elif model_type == "mlm":
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    loss, _ = model(input_ids, attention_mask, labels)
                elif model_type == "multiview":
                    views = {}
                    for view_name in batch["view_names"]:
                        views[view_name] = {
                            "input_ids": batch["views"][view_name]["input_ids"].to(device),
                            "attention_mask": batch["views"][view_name]["attention_mask"].to(device),
                        }
                    loss, _ = model(views)
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        else:
            if model_type in ["simcse", "simclr"]:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                num_views = batch["num_views"]
                loss, _ = model(input_ids, attention_mask, num_views)
            elif model_type == "tsdae":
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target_ids = batch["target_ids"].to(device)
                target_attention_mask = batch["target_attention_mask"].to(device)
                loss, _ = model(input_ids, attention_mask, target_ids, target_attention_mask)
            elif model_type == "mlm":
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                loss, _ = model(input_ids, attention_mask, labels)
            elif model_type == "multiview":
                views = {}
                for view_name in batch["view_names"]:
                    views[view_name] = {
                        "input_ids": batch["views"][view_name]["input_ids"].to(device),
                        "attention_mask": batch["views"][view_name]["attention_mask"].to(device),
                    }
                loss, _ = model(views)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        loss_meter.update(
            loss.item(),
            batch["input_ids"].size(0)
            if model_type != "multiview"
            else len(batch["item_ids"]),
        )
        pbar.set_postfix({"loss": loss_meter.avg})

        if batch_idx % 50 == 0 and device.type == "cuda":
            torch.cuda.empty_cache()

    return loss_meter.avg


def extract_embeddings(
    model,
    dataset,
    tokenizer,
    device,
    batch_size=64,  # Reduce batch size for extraction to save memory
    max_length=256,
):
    """Extract embeddings for all items."""
    model.eval()

    # Create simple dataloader
    from torch.utils.data import DataLoader  # type: ignore

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    embeddings_dict = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            item_ids = batch["item_id"]
            texts = batch["text"]

            # Tokenize
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            # Encode
            embeddings = model.encode(input_ids, attention_mask)

            # Store
            embeddings_np = embeddings.cpu().numpy()
            for i, item_id in enumerate(item_ids):
                embeddings_dict[item_id] = embeddings_np[i]

    return embeddings_dict


def train_ssl_model(
    config_path: str, model_type: str, output_dir: str
):
    """
    Train an SSL model.

    Args:
        config_path: Path to config file
        model_type: Type of model ('simcse', 'simclr', 'tsdae', 'mlm', 'multiview')
        output_dir: Output directory for checkpoints and embeddings
    """
    # Load config
    base_config = load_config("configs/base_config.yaml")
    model_config = load_config(config_path)
    config = merge_configs(base_config, model_config)

    # Set seed
    set_seed(config["split"]["random_seed"])

    # Device
    device = get_device()
    print(f"Using device: {device}")

    # Output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save config
    save_json(config, str(output_path / "config.json"))

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["backbone"])

    # Load metadata
    print("\nLoading metadata...")
    data_dir = Path(config["data"]["output_dir"])
    metadata_path = data_dir / "item_metadata.jsonl"

    metadata_dataset = ItemMetadataDataset(
        metadata_path=str(metadata_path),
        tokenizer=tokenizer,
        max_length=config["preprocessing"]["max_text_length"],
    )

    # Create dataset based on model type
    print(f"\nCreating {model_type} dataset...")

    if model_type == "simcse":
        # SimCSE with dropout augmentation
        dataset = ContrastiveDataset(
            metadata_dataset=metadata_dataset,
            augmentation_fn=simcse_dropout_augmentation,
            num_views=2,
        )
        collate_fn = lambda batch: collate_fn_contrastive(
            batch, tokenizer, config["preprocessing"]["max_text_length"]
        )

    elif model_type == "simclr":
        # SimCLR with text augmentations
        dataset = ContrastiveDataset(
            metadata_dataset=metadata_dataset,
            augmentation_fn=simclr_augmentation,
            num_views=2,
        )
        collate_fn = lambda batch: collate_fn_contrastive(
            batch, tokenizer, config["preprocessing"]["max_text_length"]
        )

    elif model_type == "tsdae":
        # TSDAE
        dataset = metadata_dataset
        deletion_prob = config.get("tsdae", {}).get("deletion_probability", 0.6)
        collate_fn = lambda batch: collate_fn_tsdae(
            batch, tokenizer, config["preprocessing"]["max_text_length"], deletion_prob
        )

    elif model_type == "mlm":
        # MLM
        dataset = metadata_dataset
        mask_prob = config.get("mlm", {}).get("mask_probability", 0.15)
        collate_fn = lambda batch: collate_fn_mlm(
            batch, tokenizer, config["preprocessing"]["max_text_length"], mask_prob
        )

    elif model_type == "multiview":
        # Multi-view contrastive
        dataset = MultiViewDataset(
            metadata_dataset=metadata_dataset,
            views=config.get("multiview", {}).get(
                "views", ["title", "description", "attributes"]
            ),
            fallback_strategy=config.get("multiview", {}).get(
                "fallback_strategy", "concatenate"
            ),
        )
        collate_fn = lambda batch: collate_fn_multiview(
            batch, tokenizer, config["preprocessing"]["max_text_length"]
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Create dataloader
    batch_size = config["training"]["batch_size"]
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Keep 0 to avoid multiprocessing memory issues
        pin_memory=False if device.type == "cpu" else True,  # Disable pin_memory on CPU
        collate_fn=collate_fn,
        persistent_workers=False,
    )

    print(f"  Dataset size: {len(dataset)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Steps per epoch: {len(dataloader)}")

    # Create model
    print(f"\nCreating {model_type} model...")

    model_kwargs = {
        "model_name": config["model"]["backbone"],
        "embedding_dim": config["model"]["embedding_dim"],
        "pooling_strategy": config["model"]["pooling_strategy"],
        "dropout": config["model"]["dropout"],
    }

    if model_type == "simcse":
        model = SimCSEModel(
            **model_kwargs, temperature=config["training"]["temperature"]
        )
    elif model_type == "simclr":
        model = SimCLRModel(
            **model_kwargs, temperature=config["training"]["temperature"]
        )
    elif model_type == "tsdae":
        model = TSDAEModel(
            **model_kwargs,
            tie_encoder_decoder=config.get("tsdae", {}).get(
                "tie_encoder_decoder", True
            ),
        )
    elif model_type == "mlm":
        model = MLMModel(**model_kwargs)
    elif model_type == "multiview":
        model = MultiViewContrastiveModel(
            **model_kwargs,
            temperature=config["training"]["temperature"],
            shared_encoder=config.get("multiview", {}).get("shared_encoder", True),
            view_names=config.get("multiview", {}).get(
                "views", ["title", "description", "attributes"]
            ),
        )

    model = model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {num_params:,}")

    # Optimizer
    print("\nCreating optimizer...")

    # Separate learning rates for backbone and projection head
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if "bert" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone_params,
                "lr": config["training"]["learning_rate_backbone"],
            },
            {"params": head_params, "lr": config["training"]["learning_rate_head"]},
        ],
        weight_decay=config["training"]["weight_decay"],
    )

    num_epochs = config["training"]["num_epochs"]
    
    # Learning rate scheduler with warmup
    num_training_steps = len(dataloader) * num_epochs
    warmup_steps = config["training"].get("warmup_steps", 0)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )
    print(f"  Learning rate scheduler: warmup={warmup_steps}, total_steps={num_training_steps}")

    # Training loop
    print(f"\n{'=' * 60}")
    print("Starting training...")
    print(f"{'=' * 60}\n")

    
    best_loss = float("inf")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)

        # Train
        avg_loss = train_epoch(model, dataloader, optimizer, scheduler, device, model_type)

        print(f"Average loss: {avg_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % config["experiment"]["checkpoint_interval"] == 0:
            checkpoint_path = output_path / f"checkpoint_epoch_{epoch + 1}.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                    "config": config,
                },
                checkpoint_path,
            )
            print(f"Checkpoint saved: {checkpoint_path}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = output_path / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                    "config": config,
                },
                best_model_path,
            )
            print(f"✓ Best model saved (loss: {best_loss:.4f})")

    # Extract embeddings
    print(f"\n{'=' * 60}")
    print("Extracting embeddings...")
    print(f"{'=' * 60}\n")

    embeddings = extract_embeddings(
        model=model,
        dataset=metadata_dataset,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
        model_type=model_type,
        max_length=config["preprocessing"]["max_text_length"],
    )

    # Save embeddings
    embeddings_path = output_path / "item_embeddings.npz"
    save_embeddings(embeddings, str(embeddings_path))
    print(f"\n✓ Embeddings saved: {embeddings_path}")
    print(f"  Total items: {len(embeddings)}")

    print(f"\n{'=' * 60}")
    print("Training completed!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SSL model")
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["simcse", "simclr", "tsdae", "mlm", "multiview"],
        help="Type of SSL model to train",
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to model config file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for checkpoints and embeddings",
    )

    args = parser.parse_args()

    train_ssl_model(
        config_path=args.config,
        model_type=args.model_type,
        output_dir=args.output_dir,
    )
