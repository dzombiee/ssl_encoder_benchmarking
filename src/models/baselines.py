"""
Baseline models for comparison.
"""

import numpy as np
from typing import Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore


class RandomEmbeddingBaseline:
    """
    Random embedding baseline.
    Each item gets a random embedding vector.
    """

    def __init__(self, embedding_dim: int = 256, seed: int = 42):
        """
        Args:
            embedding_dim: Dimension of random embeddings
            seed: Random seed
        """
        self.embedding_dim = embedding_dim
        self.seed = seed
        self.item_embeddings: Dict[str, np.ndarray] = {}

        np.random.seed(seed)

    def fit(self, item_ids: List[str]):
        """
        Create random embeddings for items.

        Args:
            item_ids: List of item IDs
        """
        for item_id in item_ids:
            # Generate random normalized embedding
            embedding = np.random.randn(self.embedding_dim)
            embedding = embedding / np.linalg.norm(embedding)  # L2 normalize
            self.item_embeddings[item_id] = embedding

        print(
            f"RandomEmbeddingBaseline: Created embeddings for {len(self.item_embeddings)} items"
        )

    def get_embedding(self, item_id: str) -> np.ndarray:
        """Get embedding for an item."""
        if item_id not in self.item_embeddings:
            # Generate on-the-fly if not seen
            embedding = np.random.randn(self.embedding_dim)
            embedding = embedding / np.linalg.norm(embedding)
            self.item_embeddings[item_id] = embedding

        return self.item_embeddings[item_id]

    def get_embeddings(self) -> Dict[str, np.ndarray]:
        """Get all item embeddings."""
        return self.item_embeddings


class TFIDFBaseline:
    """
    TF-IDF content-based baseline.
    Encodes items using TF-IDF and returns cosine similarity scores.
    """

    def __init__(
        self, max_features: int = 5000, ngram_range: tuple = (1, 2), min_df: int = 2
    ):
        """
        Args:
            max_features: Maximum number of TF-IDF features
            ngram_range: N-gram range for TF-IDF
            min_df: Minimum document frequency
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            lowercase=True,
            stop_words="english",
        )

        self.item_ids: List[str] = []
        self.tfidf_matrix = None
        self.item_to_idx: Dict[str, int] = {}

    def fit(self, train_items: List[Dict], all_items: List[Dict] = None):  # type: ignore
        """
        Fit TF-IDF on training items only, then transform all items.

        This prevents data leakage by ensuring the vocabulary is learned
        ONLY from warm/training items, not from cold/test items.

        Args:
            train_items: List of WARM item dictionaries (for learning vocabulary)
            all_items: List of ALL item dictionaries (warm + cold) to transform.
                      If None, only train_items are transformed.
        """
        if all_items is None:
            all_items = train_items

        # Extract texts
        train_texts = [item.get("full_text", "") for item in train_items]
        all_texts = [item.get("full_text", "") for item in all_items]

        # FIT on warm items only (no leakage!)
        print(
            f"TFIDFBaseline: Fitting vocabulary on {len(train_items)} WARM items only..."
        )
        self.vectorizer.fit(train_texts)

        # TRANSFORM all items (warm + cold)
        print(f"TFIDFBaseline: Transforming {len(all_items)} total items...")
        self.tfidf_matrix = self.vectorizer.transform(all_texts)

        self.item_ids = [item["parent_asin"] for item in all_items]
        self.item_to_idx = {item_id: idx for idx, item_id in enumerate(self.item_ids)}

        print(f"  Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        print(f"  TF-IDF matrix shape: {self.tfidf_matrix.shape}")  # type:ignore
        print("  ✓ No data leakage - vocabulary from warm items only!")

    def get_embedding(self, item_id: str) -> np.ndarray:
        """Get TF-IDF vector for an item."""
        if item_id not in self.item_to_idx:
            # Return zero vector for unseen items
            return np.zeros(self.tfidf_matrix.shape[1])  # type: ignore

        idx = self.item_to_idx[item_id]
        return self.tfidf_matrix[idx].toarray().flatten()  # type: ignore

    def get_embeddings(self) -> Dict[str, np.ndarray]:
        """Get all item embeddings as dense vectors."""
        embeddings = {}
        for item_id, idx in self.item_to_idx.items():
            embeddings[item_id] = self.tfidf_matrix[idx].toarray().flatten()  # type: ignore

        return embeddings

    def compute_similarity(self, item_id1: str, item_id2: str) -> float:
        """Compute cosine similarity between two items."""
        if item_id1 not in self.item_to_idx or item_id2 not in self.item_to_idx:
            return 0.0

        idx1 = self.item_to_idx[item_id1]
        idx2 = self.item_to_idx[item_id2]

        vec1 = self.tfidf_matrix[idx1]  # type: ignore
        vec2 = self.tfidf_matrix[idx2]  # type: ignore

        similarity = cosine_similarity(vec1, vec2)[0, 0]

        return similarity

    def get_most_similar(self, item_id: str, k: int = 10) -> List[tuple]:
        """
        Get k most similar items to a given item.

        Args:
            item_id: Query item ID
            k: Number of similar items to return

        Returns:
            List of (item_id, similarity_score) tuples
        """
        if item_id not in self.item_to_idx:
            return []

        idx = self.item_to_idx[item_id]
        vec = self.tfidf_matrix[idx]  # type: ignore

        # Compute similarities with all items
        similarities = cosine_similarity(vec, self.tfidf_matrix).flatten()

        # Get top k (excluding self)
        top_indices = np.argsort(similarities)[::-1][1 : k + 1]

        results = [(self.item_ids[i], similarities[i]) for i in top_indices]

        return results


class SentenceBERTBaseline:
    """
    Sentence-BERT baseline using pre-trained sentence transformers.
    Content-only baseline for comparison.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Args:
            model_name: Sentence transformer model name
        """
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError:
            raise ImportError(
                "Please install sentence-transformers: pip install sentence-transformers"
            )

        self.model = SentenceTransformer(model_name)
        self.item_ids: List[str] = []
        self.item_embeddings: Dict[str, np.ndarray] = {}

        print(f"Loaded Sentence-BERT model: {model_name}")

    def fit(self, items: List[Dict]):
        """
        Encode items using Sentence-BERT.

        Args:
            items: List of item dictionaries with 'parent_asin' and 'full_text' keys
        """
        self.item_ids = [item["parent_asin"] for item in items]
        texts = [item.get("full_text", "") for item in items]

        print(f"Encoding {len(texts)} items with Sentence-BERT...")
        embeddings = self.model.encode(
            texts, show_progress_bar=True, convert_to_numpy=True
        )

        # Store embeddings
        for item_id, embedding in zip(self.item_ids, embeddings):
            self.item_embeddings[item_id] = embedding

        print(f"  Embedding dimension: {embeddings.shape[1]}")
        print(f"  Total items: {len(self.item_embeddings)}")

    def get_embedding(self, item_id: str) -> np.ndarray:
        """Get embedding for an item."""
        return self.item_embeddings.get(
            item_id, np.zeros(384)
        )  # Default to 384-dim zero vector

    def get_embeddings(self) -> Dict[str, np.ndarray]:
        """Get all item embeddings."""
        return self.item_embeddings
