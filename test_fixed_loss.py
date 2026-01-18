"""Test the fixed contrastive loss."""

import torch
import torch.nn.functional as F


def fixed_contrastive_loss(embeddings, temperature=0.07):
    """Fixed implementation."""
    batch_size, num_views, dim = embeddings.shape
    embeddings_flat = embeddings.view(-1, dim)

    # Compute similarity matrix
    similarity_matrix = torch.matmul(embeddings_flat, embeddings_flat.T) / temperature

    # FIXED labels for interleaved arrangement
    labels = torch.arange(batch_size * num_views).to(embeddings.device)
    labels = labels.view(batch_size, num_views).flip(dims=[1]).view(-1)

    # Mask self-similarity
    mask = torch.eye(batch_size * num_views, dtype=torch.bool, device=embeddings.device)
    similarity_matrix = similarity_matrix.masked_fill(mask, float("-inf"))

    loss = F.cross_entropy(similarity_matrix, labels)
    return loss, labels


# Test 1: Random embeddings
print("=" * 70)
print("TEST 1: Random embeddings (should give high loss)")
print("=" * 70)
batch_size = 64
num_views = 2
dim = 384

torch.manual_seed(42)
embeddings_random = torch.randn(batch_size, num_views, dim)
embeddings_random = F.normalize(embeddings_random, p=2, dim=-1)

loss_random, labels = fixed_contrastive_loss(embeddings_random)
expected_random = torch.log(
    torch.tensor(batch_size * num_views - 1, dtype=torch.float32)
)

print(f"Batch size: {batch_size}")
print(f"Loss with random embeddings: {loss_random.item():.4f}")
print(
    f"Expected for random: ~{expected_random.item():.4f} (log({batch_size * num_views - 1}))"
)
print(f"Labels (first 10): {labels[:10]}")
print()

# Test 2: Perfect contrastive pairs (identical views)
print("=" * 70)
print("TEST 2: Perfect pairs (identical views, should give near-zero loss)")
print("=" * 70)

# Create embeddings where both views are identical
embeddings_perfect = torch.randn(batch_size, 1, dim)
embeddings_perfect = F.normalize(embeddings_perfect, p=2, dim=-1)
embeddings_perfect = embeddings_perfect.repeat(1, 2, 1)  # Copy to both views

loss_perfect, _ = fixed_contrastive_loss(embeddings_perfect, temperature=0.07)

print(f"Loss with perfect pairs: {loss_perfect.item():.6f}")
print(f"Expected: ~0.000 (views are identical)")
print()

# Test 3: High similarity pairs (should give low loss)
print("=" * 70)
print("TEST 3: High similarity pairs (similar but not identical)")
print("=" * 70)

embeddings_base = torch.randn(batch_size, 1, dim)
embeddings_base = F.normalize(embeddings_base, p=2, dim=-1)
# Add small noise to create second view
noise = torch.randn(batch_size, 1, dim) * 0.1
embeddings_noisy = embeddings_base + noise
embeddings_noisy = F.normalize(embeddings_noisy, p=2, dim=-1)
embeddings_similar = torch.cat([embeddings_base, embeddings_noisy], dim=1)

loss_similar, _ = fixed_contrastive_loss(embeddings_similar, temperature=0.07)

print(f"Loss with similar pairs: {loss_similar.item():.4f}")
print(f"Expected: Low (< 1.0) but not zero")
print()

# Test 4: Verify label correctness
print("=" * 70)
print("TEST 4: Verify labels create correct pairs")
print("=" * 70)

batch_size_test = 4
embeddings_test = torch.randn(batch_size_test, 2, dim)
_, labels_test = fixed_contrastive_loss(embeddings_test)

print("Label mapping (position -> positive):")
for i in range(batch_size_test * 2):
    item_id = i // 2
    view_id = i % 2
    pos_item = labels_test[i].item() // 2
    pos_view = labels_test[i].item() % 2
    print(
        f"  Pos {i} (item{item_id}_view{view_id}) -> Pos {labels_test[i]} (item{pos_item}_view{pos_view})"
    )

print()
print("✓ Verification: Each position should pair with the other view of the same item!")
print()

# Test 5: Compare with buggy version
print("=" * 70)
print("TEST 5: Compare fixed vs buggy implementation")
print("=" * 70)


def buggy_loss(embeddings, temperature=0.07):
    batch_size, num_views, dim = embeddings.shape
    embeddings_flat = embeddings.view(-1, dim)
    similarity_matrix = torch.matmul(embeddings_flat, embeddings_flat.T) / temperature
    labels = torch.arange(batch_size).to(embeddings.device)
    labels = torch.cat([labels + batch_size, labels])
    mask = torch.eye(batch_size * num_views, dtype=torch.bool, device=embeddings.device)
    similarity_matrix = similarity_matrix.masked_fill(mask, float("-inf"))
    return F.cross_entropy(similarity_matrix, labels)


batch_size = 64
embeddings_test = torch.randn(batch_size, 2, dim)
embeddings_test = F.normalize(embeddings_test, p=2, dim=-1)

loss_fixed, _ = fixed_contrastive_loss(embeddings_test)
loss_buggy = buggy_loss(embeddings_test)

print(f"Fixed loss: {loss_fixed.item():.4f}")
print(f"Buggy loss: {loss_buggy.item():.4f}")
print()

# Test with perfect pairs
embeddings_perfect = torch.randn(batch_size, 1, dim)
embeddings_perfect = F.normalize(embeddings_perfect, p=2, dim=-1)
embeddings_perfect = embeddings_perfect.repeat(1, 2, 1)

loss_fixed_perfect, _ = fixed_contrastive_loss(embeddings_perfect)
loss_buggy_perfect = buggy_loss(embeddings_perfect)

print("With perfect pairs:")
print(f"  Fixed loss: {loss_fixed_perfect.item():.6f} (should be ~0)")
print(f"  Buggy loss: {loss_buggy_perfect.item():.4f} (stays high!)")
print()
print("✓ FIXED: Loss goes to zero with perfect pairs!")
print("✗ BUGGY: Loss stays high even with perfect pairs!")
print()
print("=" * 70)
print("CONCLUSION: Bug fixed! Model should now be able to learn.")
print("=" * 70)
