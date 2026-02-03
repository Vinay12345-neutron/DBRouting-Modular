# High-Performance Setup (RTX 4090 / 24GB VRAM)

Since you have access to an RTX 4090, you can run a much more powerful model than the "Nano" 0.5B version. This will significantly improve the accuracy of the **Modular Reasoning** re-ranking.

### 1. Upgrade the LLM
Open `reranker.py` and change line 18:

```python
# Change from 0.5B to 14B or 32B
LLM_MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
# OR "mistralai/Mistral-7B-Instruct-v0.3" 
```

### 2. Performance Tuning
On a 4090, you can increase processing speed.
*   **Batch Size**: In `retrieval.py`, you can increase `batch_size` from 1 to 16 or 32 for much faster embedding generation.
*   **Precision**: Ensure `torch_dtype=torch.float16` is used (already set); the 4090 handles this natively with Tensor Cores.

### 3. Expected Improvements
*   **Better Mapping**: Larger models are better at identifying that "revenue" in a query maps to "total_sales" in a schema.
*   **Smarter Adjacency**: Adjacency lists will be more accurate for complex joins (4+ tables).
*   **Recall@1**: You should see a significant jump in accuracy compared to the 0.5B baseline.

### 4. Running the Pro Pipeline
1.  **Fresh Environment**:
    ```bash
    python -m venv venv
    .\venv\Scripts\Activate
    # EXTREMELY IMPORTANT: Install CUDA-enabled Torch
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    pip install -r requirements.txt
    ```
2.  Ensure `data/` is present. (I think processed data works, as its already processed, no need the official zip folder.)
2.  `python prepare_data.py`
3.  `python retrieval.py`
4.  `python reranker.py` (Wait for it to download the 14B model)
5.  `python evaluate.py`
