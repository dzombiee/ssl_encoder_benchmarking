"""
Text augmentation functions for contrastive learning.
"""

import random
from typing import List


def token_dropout(text: str, dropout_prob: float = 0.1) -> str:
    """Randomly drop tokens from text."""
    words = text.split()
    if len(words) <= 2:
        return text

    kept_words = [w for w in words if random.random() > dropout_prob]
    if len(kept_words) == 0:  # Keep at least one word
        kept_words = [random.choice(words)]

    return " ".join(kept_words)


def token_shuffle(text: str, window_size: int = 3) -> str:
    """Shuffle tokens within local windows."""
    words = text.split()
    if len(words) <= 2:
        return text

    shuffled = words.copy()
    for i in range(0, len(words), window_size):
        window = shuffled[i : min(i + window_size, len(words))]
        random.shuffle(window)
        shuffled[i : min(i + window_size, len(words))] = window

    return " ".join(shuffled)


def random_crop(
    text: str, crop_ratio_min: float = 0.7, crop_ratio_max: float = 0.95
) -> str:
    """Randomly crop a portion of the text."""
    words = text.split()
    if len(words) <= 3:
        return text

    crop_ratio = random.uniform(crop_ratio_min, crop_ratio_max)
    crop_length = max(3, int(len(words) * crop_ratio))

    if crop_length >= len(words):
        return text

    start_idx = random.randint(0, len(words) - crop_length)
    cropped = words[start_idx : start_idx + crop_length]

    return " ".join(cropped)


def random_swap(text: str, num_swaps: int = 2) -> str:
    """Randomly swap positions of words."""
    words = text.split()
    if len(words) <= 2:
        return text

    for _ in range(min(num_swaps, len(words) // 2)):
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]

    return " ".join(words)


def compose_augmentations(augmentation_types: List[str], **kwargs):
    """
    Create a composition of augmentation functions.

    Args:
        augmentation_types: List of augmentation names
        **kwargs: Parameters for each augmentation

    Returns:
        Augmentation function
    """
    aug_map = {
        "token_dropout": lambda t: token_dropout(t, kwargs.get("dropout_prob", 0.1)),
        "token_shuffle": lambda t: token_shuffle(t, kwargs.get("shuffle_window", 3)),
        "random_crop": lambda t: random_crop(
            t, kwargs.get("crop_ratio_min", 0.7), kwargs.get("crop_ratio_max", 0.95)
        ),
        "random_swap": lambda t: random_swap(t, kwargs.get("num_swaps", 2)),
    }

    def augment(text: str) -> str:
        """Apply random augmentation from the list."""
        if not augmentation_types:
            return text

        # Randomly select one augmentation
        aug_type = random.choice(augmentation_types)
        aug_fn = aug_map.get(aug_type)

        if aug_fn:
            return aug_fn(text)
        return text

    return augment


def simcse_dropout_augmentation(text: str) -> str:
    """
    SimCSE augmentation: just return the same text.
    Dropout happens in the model (different dropout masks create different views).
    """
    return text


def simclr_augmentation(text: str) -> str:
    """
    SimCLR-style augmentation: apply multiple augmentations.
    """
    # Apply 1-2 random augmentations
    num_augs = random.randint(1, 2)

    aug_fns = [
        lambda t: token_dropout(t, 0.1),
        lambda t: random_crop(t, 0.7, 0.9),
        lambda t: token_shuffle(t, 3),
    ]

    selected_augs = random.sample(aug_fns, num_augs)

    augmented = text
    for aug_fn in selected_augs:
        augmented = aug_fn(augmented)

    return augmented
