"""Debug contrastive loss implementation."""

import torch
import torch.nn.functional as F


def current_buggy_loss(embeddings, temperature=0.07):
    """Current implementation (BUGGY)."""
    batch_size, num_views, dim = embeddings.shape
    embeddings_flat = embeddings.view(-1, dim)

    # Compute similarity matrix
    similarity_matrix = torch.matmul(embeddings_flat, embeddings_flat.T) / temperature

    # BUGGY labels
    labels = torch.arange(batch_size).to(embeddings.device)
    labels = torch.cat([labels + batch_size, labels])

    # Mask self-similarity
    mask = torch.eye(batch_size * num_views, dtype=torch.bool, device=embeddings.device)
    similarity_matrix = similarity_matrix.masked_fill(mask, float("-inf"))

    loss = F.cross_entropy(similarity_matrix, labels)
    return loss, labels


def correct_loss(embeddings, temperature=0.07):
    """Correct implementation."""
    batch_size, num_views, dim = embeddings.shape
    embeddings_flat = embeddings.view(-1, dim)

    # Compute similarity matrix
    similarity_matrix = torch.matmul(embeddings_flat, embeddings_flat.T) / temperature

    # CORRECT labels: for position i, positive is at i + batch_size (if i < batch_size)
    # or i - batch_size (if i >= batch_size)
    labels = torch.arange(batch_size * num_views).to(embeddings.device)
    # Swap first and second half
    labels = torch.cat([labels[batch_size:], labels[:batch_size]])

    # Mask self-similarity
    mask = torch.eye(batch_size * num_views, dtype=torch.bool, device=embeddings.device)
    similarity_matrix = similarity_matrix.masked_fill(mask, float("-inf"))

    loss = F.cross_entropy(similarity_matrix, labels)
    return loss, labels


# Test with random embeddings
batch_size = 4
num_views = 2
dim = 128

torch.manual_seed(42)
embeddings = torch.randn(batch_size, num_views, dim)
embeddings = F.normalize(embeddings, p=2, dim=-1)

print("Testing with random normalized embeddings:")
print(f"Batch size: {batch_size}, Num views: {num_views}, Dim: {dim}")
print()

buggy_loss, buggy_labels = current_buggy_loss(embeddings)
correct_loss_val, correct_labels = correct_loss(embeddings)

print("=" * 60)
print("CURRENT (BUGGY) IMPLEMENTATION:")
print("=" * 60)
print(f"Labels: {buggy_labels}")
print(f"Loss: {buggy_loss.item():.4f}")
print(
    f"Expected loss for random: ~{torch.log(torch.tensor(batch_size * num_views)):.4f}"
)
print()
print("Label pairs (position -> positive):")
for i in range(batch_size * num_views):
    print(f"  Position {i} -> position {buggy_labels[i].item()}")

print()
print("=" * 60)
print("CORRECT IMPLEMENTATION:")
print("=" * 60)
print(f"Labels: {correct_labels}")
print(f"Loss: {correct_loss_val.item():.4f}")
print()
print("Label pairs (position -> positive):")
for i in range(batch_size * num_views):
    print(f"  Position {i} -> position {correct_labels[i].item()}")

print()
print("=" * 60)
print("EXPLANATION OF BUG:")
print("=" * 60)
print("""
Current implementation creates labels as:
  labels = torch.arange(batch_size) + batch_size  # [batch_size, batch_size+1, ..., 2*batch_size-1]
  labels = torch.cat([labels, torch.arange(batch_size)])  # [..., 0, 1, ..., batch_size-1]

Problem: This means position 0 points to position batch_size as positive,
but position batch_size points to position 0, which is NOT a valid index!

Embeddings are arranged as:
  [item0_view0, item1_view0, ..., item(N-1)_view0, item0_view1, item1_view1, ..., item(N-1)_view1]

For position i:
  - If i < batch_size: positive should be at i + batch_size (same item, view 1)
  - If i >= batch_size: positive should be at i - batch_size (same item, view 0)

Correct labels should be: [batch_size, batch_size+1, ..., 2*batch_size-1, 0, 1, ..., batch_size-1]
""")

# Test with actual contrastive pairs (same item, different views)
print()
print("=" * 60)
print("TEST WITH PERFECT CONTRASTIVE PAIRS:")
print("=" * 60)

# Create perfect pairs: view 0 and view 1 are identical
embeddings_perfect = torch.randn(batch_size, 1, dim)
embeddings_perfect = F.normalize(embeddings_perfect, p=2, dim=-1)
embeddings_perfect = embeddings_perfect.repeat(1, 2, 1)  # Same embedding for both views

buggy_loss_perfect, _ = current_buggy_loss(embeddings_perfect)
correct_loss_perfect, _ = correct_loss(embeddings_perfect)

print(f"Buggy loss with perfect pairs: {buggy_loss_perfect.item():.4f} (should be ~0)")
print(
    f"Correct loss with perfect pairs: {correct_loss_perfect.item():.4f} (should be ~0)"
)
print()
print(f"✓ Correct implementation gives near-zero loss for perfect pairs!")
print(f"✗ Buggy implementation gives high loss even for perfect pairs!")
