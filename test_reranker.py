"""
Diagnostic script to identify reranker issues
Run this first to see where the problem is
"""

import sys
print("=" * 60)
print("RERANKER DIAGNOSTIC TOOL")
print("=" * 60)

# Test 1: Python imports
print("\n[1/7] Testing Python imports...")
try:
    import os
    import json
    import gc
    print("  ✓ Standard library imports OK")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Test 2: Torch
print("\n[2/7] Testing PyTorch...")
try:
    import torch
    print(f"  ✓ PyTorch: {torch.__version__}")
    print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  ✓ CUDA version: {torch.version.cuda}")
        print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  ⚠ WARNING: CUDA not available, will run on CPU (very slow)")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Test 3: Transformers
print("\n[3/7] Testing Transformers...")
try:
    import transformers
    print(f"  ✓ Transformers: {transformers.__version__}")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Test 4: bitsandbytes (for 8-bit quantization)
print("\n[4/7] Testing bitsandbytes...")
try:
    import bitsandbytes as bnb
    print(f"  ✓ BitsAndBytes: {bnb.__version__}")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    print("  ⚠ 8-bit quantization will not work")
    print("  Run: pip install bitsandbytes")

# Test 5: Other dependencies
print("\n[5/7] Testing other dependencies...")
try:
    import numpy as np
    print(f"  ✓ NumPy: {np.__version__}")
    from tqdm import tqdm
    print(f"  ✓ tqdm: OK")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Test 6: File structure
print("\n[6/7] Checking file structure...")
files_to_check = [
    "processed_data/spider_route_test.json",
    "processed_data/bird_route_test.json",
    "results/spider_retrieval_results.json",
    "results/bird_retrieval_results.json",
]

for f in files_to_check:
    if os.path.exists(f):
        size_mb = os.path.getsize(f) / (1024*1024)
        print(f"  ✓ {f} ({size_mb:.1f} MB)")
    else:
        print(f"  ✗ MISSING: {f}")

# Test 7: Quick model download test
print("\n[7/7] Testing model download (this may take 1-2 minutes)...")
print("  Attempting to load tokenizer...")
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        trust_remote_code=True
    )
    print("  ✓ Tokenizer loaded successfully")
    print("  ✓ Model files are accessible")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    print("  This is the likely cause of the hang!")
    print("  Possible fixes:")
    print("    1. Check internet connection")
    print("    2. Clear HuggingFace cache: rm -rf ~/.cache/huggingface")
    print("    3. Try: export HF_ENDPOINT=https://hf-mirror.com")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
print("\nIf all tests passed, the script should work.")
print("If any test failed, fix that issue first.")
