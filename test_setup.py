"""
Quick test script to verify the pipeline setup.
Tests data loading, model creation, and basic functionality.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import torch
from transformers import AutoTokenizer
from data.preprocessing import load_jsonl
from models.simcse import SimCSEModel
from models.tsdae import TSDAEModel
from models.mlm import MLMModel
from models.multiview import MultiViewContrastiveModel
from models.baselines import RandomEmbeddingBaseline, TFIDFBaseline


def test_data_loading():
    """Test data loading."""
    print("\n" + "="*60)
    print("Testing Data Loading...")
    print("="*60)
    
    # Test loading JSONL
    try:
        reviews = load_jsonl('data/All_Beauty.jsonl')
        print(f"✓ Loaded {len(reviews)} reviews")
        print(f"  Sample review: {reviews[0]['user_id']}")
    except Exception as e:
        print(f"✗ Failed to load reviews: {e}")
        return False
    
    try:
        metadata = load_jsonl('data/meta_All_Beauty.jsonl')
        print(f"✓ Loaded {len(metadata)} items")
        print(f"  Sample item: {metadata[0]['parent_asin']}")
    except Exception as e:
        print(f"✗ Failed to load metadata: {e}")
        return False
    
    return True


def test_model_creation():
    """Test creating all model types."""
    print("\n" + "="*60)
    print("Testing Model Creation...")
    print("="*60)
    
    device = torch.device('cpu')
    
    try:
        # SimCSE
        model = SimCSEModel(
            model_name='distilbert-base-uncased',
            embedding_dim=128,
            pooling_strategy='mean'
        )
        print(f"✓ SimCSE model created")
        
        # TSDAE
        model = TSDAEModel(
            model_name='distilbert-base-uncased',
            embedding_dim=128
        )
        print(f"✓ TSDAE model created")
        
        # MLM
        model = MLMModel(
            model_name='distilbert-base-uncased',
            embedding_dim=128
        )
        print(f"✓ MLM model created")
        
        # Multi-view
        model = MultiViewContrastiveModel(
            model_name='distilbert-base-uncased',
            embedding_dim=128
        )
        print(f"✓ Multi-view model created")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False


def test_baselines():
    """Test baseline models."""
    print("\n" + "="*60)
    print("Testing Baseline Models...")
    print("="*60)
    
    try:
        # Random baseline
        random_baseline = RandomEmbeddingBaseline(embedding_dim=128)
        random_baseline.fit(['item1', 'item2', 'item3'])
        emb = random_baseline.get_embedding('item1')
        print(f"✓ Random baseline created (embedding shape: {emb.shape})")
        
        # TF-IDF baseline
        items = [
            {'parent_asin': 'item1', 'full_text': 'this is a product description'},
            {'parent_asin': 'item2', 'full_text': 'another product with features'},
            {'parent_asin': 'item3', 'full_text': 'beauty product for testing'}
        ]
        tfidf_baseline = TFIDFBaseline(max_features=100)
        tfidf_baseline.fit(items)
        emb = tfidf_baseline.get_embedding('item1')
        print(f"✓ TF-IDF baseline created (embedding shape: {emb.shape})")
        
        return True
        
    except Exception as e:
        print(f"✗ Baseline creation failed: {e}")
        return False


def test_tokenizer():
    """Test tokenizer."""
    print("\n" + "="*60)
    print("Testing Tokenizer...")
    print("="*60)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        
        text = "This is a test product description"
        encoded = tokenizer(text, padding=True, truncation=True, max_length=128)
        
        print(f"✓ Tokenizer loaded")
        print(f"  Input: '{text}'")
        print(f"  Tokens: {len(encoded['input_ids'])}")
        
        return True
        
    except Exception as e:
        print(f"✗ Tokenizer test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("   SSL Cold-Start Recommendation - System Test")
    print("="*70)
    
    results = []
    
    # Run tests
    results.append(("Data Loading", test_data_loading()))
    results.append(("Tokenizer", test_tokenizer()))
    results.append(("Model Creation", test_model_creation()))
    results.append(("Baselines", test_baselines()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n" + "="*60)
        print("✓ All tests passed! System is ready.")
        print("="*60)
        print("\nNext steps:")
        print("  1. Run: python src/data/preprocessing.py")
        print("  2. Run: bash run_pipeline.sh")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("✗ Some tests failed. Please check the errors above.")
        print("="*60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
