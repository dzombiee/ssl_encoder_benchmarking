"""Verify how views are arranged after collate_fn."""

import torch

# Simulate collate_fn_contrastive behavior
batch_size = 4
num_views = 2

# In collate_fn, we do: all_views.extend(item["views"])
# This means for each item, we add [view0, view1] sequentially
# Result: [item0_view0, item0_view1, item1_view0, item1_view1, ...]

print("How views are arranged after collate_fn:")
print("=" * 60)
all_views = []
for i in range(batch_size):
    for v in range(num_views):
        all_views.append(f"item{i}_view{v}")

for idx, view in enumerate(all_views):
    print(f"Position {idx}: {view}")

print()
print("Current label construction:")
print("=" * 60)
labels = torch.arange(batch_size)
labels = torch.cat([labels + batch_size, labels])
print(f"Labels: {labels}")

print()
print("Mapping (position -> its positive):")
print("=" * 60)
for i, label in enumerate(labels):
    if i < len(all_views):
        print(
            f"Position {i} ({all_views[i]}) -> position {label} ({all_views[label] if label < len(all_views) else 'OUT OF BOUNDS!'})"
        )

print()
print("=" * 60)
print("THE BUG IS CLEAR:")
print("=" * 60)
print("""
The embeddings are interleaved: [item0_v0, item0_v1, item1_v0, item1_v1, ...]

But the loss expects them grouped: [item0_v0, item1_v0, ..., item0_v1, item1_v1, ...]

Position 0 (item0_view0) should pair with position 1 (item0_view1)
Position 1 (item0_view1) should pair with position 0 (item0_view0)
Position 2 (item1_view0) should pair with position 3 (item1_view1)
etc.

Current labels point to wrong positions completely!
""")

print()
print("CORRECT labels for interleaved arrangement:")
print("=" * 60)
# For interleaved: position i pairs with i+1 if i is even, i-1 if i is odd

correct_labels_list = []
for i in range(batch_size * num_views):
    if i % 2 == 0:
        correct_labels_list.append(i + 1)
    else:
        correct_labels_list.append(i - 1)

correct_labels = torch.tensor(correct_labels_list)
print(f"Correct labels: {correct_labels}")

print()
print("Correct mapping:")
for i, label in enumerate(correct_labels):
    print(f"Position {i} ({all_views[i]}) -> position {label} ({all_views[label]})")
