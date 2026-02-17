"""
Dense Retrieval Script for Database Query Routing (Spider & BIRD)

This script performs the initial dense retrieval stage for cross-domain
NL-to-SQL query routing experiments using a pretrained embedding model.

Overview:
- Encodes natural language queries and database schemas into vector embeddings
  using a sentence-level embedding model.
- Computes cosine similarity between query embeddings and schema embeddings.
- Retrieves the top-K most relevant database schemas per query.
- Produces retrieval outputs used as input for downstream re-ranking stages.

Key Characteristics:
- Uses a pretrained embedding model without any fine-tuning or training.
- All computation is performed locally (GPU preferred).
- Retrieval is schema-level (database selection), not SQL generation.
- Designed for reproducible baselines and ablation studies.

Input:
- Processed routing datasets:
  - `processed_data/spider_route_test.json`
  - `processed_data/bird_route_test.json`
- Raw schema definitions loaded from `tables.json` files.

Output:
- Retrieval results saved as JSON:
  - `results/spider_retrieval_results.json`
  - `results/bird_retrieval_results.json`
- Each entry contains the query, gold database ID, and a ranked list of
  retrieved candidate databases.

Supported Datasets:
- Spider
- BIRD (BirdSQL)

Notes:
- CUDA is strongly recommended for feasible runtime.
- Embeddings are L2-normalized to enable cosine similarity via dot product.
- This script represents the first stage of a retrieval → re-ranking pipeline
  for database query routing.
"""

import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
import pandas as pd
from typing import List, Dict
from dotenv import load_dotenv

# Load key-value pairs from .env file (e.g., HF_TOKEN)
load_dotenv()

# Configuration Constants
DATA_DIR = "processed_data"
RAW_DATA_DIR = "data"
# User requested "Qwen3-Embedding-4B-Instruct"
MODEL_NAME = "Qwen/Qwen3-Embedding-4B" 
OUTPUT_DIR = "results"
METRICS_FILE = os.path.join(OUTPUT_DIR, "baseline_metrics.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_device() -> str:
    """
    Determines the computation device.
    Prioritizes CUDA. Raises a warning/error if CUDA is expected but missing.
    """
    if torch.cuda.is_available():
        return "cuda"
    
    print("WARNING: CUDA is not available. Running on CPU. This will be very slow and may OOM.")
    return "cpu"

class EmbeddingModel:
    """
    Wrapper for the Qwen-Embedding model handling tokenization,
    quantization (for low VRAM), and embedding generation.
    """
    def __init__(self, model_name: str = MODEL_NAME):
        self.device = get_device()
        print(f"Loading model {model_name} on {self.device}...")
        
        # Configure 4-bit quantization to fit ~12GB model into ~4-6GB VRAM
        # Update: Qwen-0.6B is small (~1GB), so we disable quantization to avoid complexity/overhead.
        bnb_config = None
        # if self.device == "cuda":
        #    print("Using 4-bit quantization (NF4)...")
        #    bnb_config = BitsAndBytesConfig(...)
        
        # Load Tokenizer
        # use_fast=True is default, but sometimes False is more stable if OOM occurs on loading
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Load Model
        # Removing offload_folder to avoid excessive System RAM usage (OS Error 1455)
        # We rely on 0.6B size fitting easily.
        self.model = AutoModel.from_pretrained(
            model_name, 
            trust_remote_code=True,
            quantization_config=bnb_config,
            # device_map="auto", # Changed to explicit due to OS 1455
            device_map={"": 0} if self.device == "cuda" else None,
            low_cpu_mem_usage=True, # Re-enabled for 0.6B to save RAM
            torch_dtype=torch.float16 # FP16 is standard for 20/30 series
        )
            
        self.model.eval()

    def encode(self, texts: List[str], batch_size: int = 1) -> np.ndarray:
        """
        Generates embeddings for a list of texts.
        
        Args:
            texts: List of strings to encode.
            batch_size: Number of texts to process at once. Keep low (1) for low VRAM/System RAM.
            
        Returns:
            Numpy array of embeddings of shape (N, D).
        """
        embeddings = []
        # Process in batches to manage memory
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch_texts = texts[i : i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors="pt"
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                last_hidden = outputs.last_hidden_state
                
                # Mean Pooling (Standard provider-agnostic approach)
                attention_mask = inputs.attention_mask
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                batch_embeddings = sum_embeddings / sum_mask
                
                # Normalize embeddings (Cosine similarity requires normalized vectors)
                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                embeddings.append(batch_embeddings.cpu().numpy())
                
        if not embeddings:
            return np.array([])
            
        return np.concatenate(embeddings, axis=0)

def load_schemas() -> Dict[str, str]:
    """
    Parses `tables.json` files from Spider/Bird to create text representations of schemas.
    
    Returns:
        Dictionary mapping db_id to schema text description.
    """
    schemas = {}
    
    # Define paths to potential schema files
    files = [
        # Spider - corrected paths
        os.path.join(DATA_DIR, "spider_tables.json"),      # Main Spider schemas
        os.path.join(DATA_DIR, "spider_test_tables.json"), # Test-specific schemas
        os.path.join(DATA_DIR, "tables.json"),             # Fallback
        os.path.join(RAW_DATA_DIR, "spider_data", "tables.json"),
        
        # Bird
        os.path.join(DATA_DIR, "dev_tables.json"),      # Bird Dev
        os.path.join(DATA_DIR, "train_tables.json"),    # Bird Train
        os.path.join(RAW_DATA_DIR, "train", "train_tables.json"),
    ]
    
    print(f"Searching for schemas in: {files}")
    for f_path in files:
        if os.path.exists(f_path):
            print(f"Reading {f_path}...")
            with open(f_path, 'r') as f:
                content = json.load(f)
                for db in content:
                    db_id = db['db_id']
                    
                    # Construct clean text representation:
                    # "Table: [name], Columns: [col1, col2, ...]"
                    schema_text = []
                    table_names = db['table_names_original']
                    column_names = db['column_names_original'] # List of [table_idx, name]
                    
                    # Group columns by table index
                    cols_by_table = {i: [] for i in range(len(table_names))}
                    for table_idx, col_name in column_names:
                        if table_idx >= 0: # -1 indicates '*'
                            cols_by_table[table_idx].append(col_name)
                            
                    for i, table in enumerate(table_names):
                        cols_str = ", ".join(cols_by_table[i])
                        schema_text.append(f"Table: {table}, Columns: {cols_str}")
                        
                    full_text = f"Database: {db_id}. " + "; ".join(schema_text)
                    
                    # Add Foreign Keys (Critical for Spider)
                    fks = db.get('foreign_keys', [])
                    if fks:
                        fk_list = []
                        for fk_pair in fks:
                            # fk_pair is [col_idx_from, col_idx_to]
                            if len(fk_pair) == 2:
                                c_from_idx, c_to_idx = fk_pair
                                # column_names contains [table_idx, col_name]
                                if c_from_idx < len(column_names) and c_to_idx < len(column_names):
                                    t_idx_from, c_name_from = column_names[c_from_idx]
                                    t_idx_to, c_name_to = column_names[c_to_idx]
                                    
                                    if t_idx_from < len(table_names) and t_idx_to < len(table_names):
                                        t_from = table_names[t_idx_from]
                                        t_to = table_names[t_idx_to]
                                        fk_list.append(f"{t_from}.{c_name_from} -> {t_to}.{c_name_to}")
                        
                        if fk_list:
                            full_text += ". Foreign keys: " + ", ".join(fk_list)
                            
                    schemas[db_id] = full_text
    
    print(f"Loaded {len(schemas)} unique database schemas.")
    return schemas

def run_retrieval(model: EmbeddingModel, queries: List[str], db_ids: List[str], schemas: Dict[str, str], k: int = 5) -> List[List[str]]:
    """
    Performs dense retrieval using cosine similarity.
    
    Args:
        model: Loaded EmbeddingModel.
        queries: List of natural language requests.
        db_ids: (Unused for retrieval logic, but good for reference) Gold DB IDs.
        schemas: Dictionary of all available DB schemas.
        k: Number of results to retrieve.
        
    Returns:
        List of lists, where each inner list contains top-K retrieved db_ids.
    """
    # 1. Encode Schemas (Candidate DBs)
    unique_db_ids = list(schemas.keys())
    unique_schema_texts = [schemas[db_id] for db_id in unique_db_ids]
    
    print(f"Encoding {len(unique_schema_texts)} Key Schemas (Repository)...")
    # Batch size 1 for safety
    schema_embeds = model.encode(unique_schema_texts, batch_size=1) 
    
    # 2. Encode Queries
    print(f"Encoding {len(queries)} Queries...")
    query_embeds = model.encode(queries, batch_size=1)
    
    # 3. Compute Similarity & Retrieve
    print("Computing Similarity Matrix...")
    # Using chunked matrix multiplication to avoid OOM with large N*M matrix
    all_top_k_dbs = []
    
    chunk_size = 100
    for i in tqdm(range(0, len(query_embeds), chunk_size), desc="Retrieving"):
        q_chunk = torch.tensor(query_embeds[i:i+chunk_size]).to(model.model.device) # [C, D]
        # Ensure schema embeddings are on the same device
        s_matrix = torch.tensor(schema_embeds).to(model.model.device) # [M, D]
        
        # Dot product (Cosine sim since normalized)
        scores = torch.mm(q_chunk, s_matrix.t()) # [C, M]
        topk_scores, topk_indices = torch.topk(scores, k=k, dim=1)
        
        topk_indices = topk_indices.detach().cpu().numpy()
        
        for idx_row in topk_indices:
            retrieved_dbs = [unique_db_ids[idx] for idx in idx_row]
            all_top_k_dbs.append(retrieved_dbs)
            
        # Clear VRAM cache after chunk
        del q_chunk, s_matrix, scores, topk_scores
        torch.cuda.empty_cache()
            
    return all_top_k_dbs

def calculate_metrics(dataset_name, queries, gold_dbs, top_k_results):
    k_values = [1, 3, 5, 10, 20]
    recalls = {k: [] for k in k_values}
    # MAP (Mean Average Precision) = MRR (Mean Reciprocal Rank) in this context 
    # since there is only 1 relevant item (gold_db) per query.
    mrr_sum = 0
    
    for gold, retrieved in zip(gold_dbs, top_k_results):
        # Recall @ K
        for k in k_values:
            recalls[k].append(1 if gold in retrieved[:k] else 0)
            
        # MRR / MAP calculation
        if gold in retrieved:
            rank = retrieved.index(gold) + 1
            mrr_sum += 1.0 / rank
            
    # Aggregate
    num_samples = len(gold_dbs)
    metrics = {}
    for k in k_values:
        metrics[f"R@{k}"] = np.mean(recalls[k])
        
    metrics["MAP"] = mrr_sum / num_samples
    metrics["Dataset"] = dataset_name
    metrics["Model"] = MODEL_NAME
    
    print(f"\n--- {dataset_name} Metrics (N={num_samples}) ---")
    print(f"R@1: {metrics['R@1']:.4f}")
    print(f"R@3: {metrics['R@3']:.4f}")
    print(f"R@5: {metrics['R@5']:.4f}")
    print(f"R@10:{metrics['R@10']:.4f}")
    print(f"R@20:{metrics['R@20']:.4f}")
    print(f"MAP: {metrics['MAP']:.4f}")
            
    return metrics

def save_metrics_to_csv(metrics_list):
    df = pd.DataFrame(metrics_list)
    # Check if file exists to determine if we write header
    mode = 'a' if os.path.exists(METRICS_FILE) else 'w'
    header = not os.path.exists(METRICS_FILE)
    df.to_csv(METRICS_FILE, mode=mode, header=header, index=False)
    print(f"\nMetrics appended to {METRICS_FILE}")

def main():
    try:
        model = EmbeddingModel()
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return

    schemas = load_schemas()
    if not schemas:
        print("No schemas loaded. Check data/ path.")
        return
        
    all_metrics = []

    # Process Spider-Route
    spider_path = os.path.join(DATA_DIR, "spider_route_test.json")
    if os.path.exists(spider_path):
        print("\n=== Processing Spider-Route ===")
        try:
            with open(spider_path, 'r') as f:
                spider_test = json.load(f)
            
            # Using subset logic if needed (currently full)
            spider_queries = [item['question'] for item in spider_test]
            spider_gold_dbs = [item['db_id'] for item in spider_test]
            
            top_k_results = run_retrieval(model, spider_queries, spider_gold_dbs, schemas, k=20)
            
            results = []
            for q, g, r in zip(spider_queries, spider_gold_dbs, top_k_results):
                results.append({"question": q, "gold_db": g, "retrieved_dbs": r})
            
            with open(os.path.join(OUTPUT_DIR, "spider_retrieval_results.json"), 'w') as f:
                json.dump(results, f, indent=2)
                
            metrics = calculate_metrics("Spider", spider_queries, spider_gold_dbs, top_k_results)
            all_metrics.append(metrics)

            # Save Misclassified (Gold not in Top 1)
            misclassified = []
            for q, g, r in zip(spider_queries, spider_gold_dbs, top_k_results):
                if g != r[0]:
                    try:
                        rank = r.index(g) + 1
                    except ValueError:
                        rank = -1 # Not in Top-K
                    
                    misclassified.append({
                        "question": q,
                        "gold_db": g,
                        "retrieved_top5": r[:5],
                        "gold_rank": rank
                    })
            
            with open(os.path.join(OUTPUT_DIR, "spider_misclassified.json"), 'w') as f:
                json.dump(misclassified, f, indent=2)
            print(f"Saved {len(misclassified)} misclassified queries to spider_misclassified.json")
            
        except Exception as e:
            print(f"Error processing Spider: {e}")

    # Process Bird-Route
    bird_path = os.path.join(DATA_DIR, "bird_route_test.json")
    if os.path.exists(bird_path):
        print("\n=== Processing Bird-Route ===")
        try:
            with open(bird_path, 'r') as f:
                bird_test = json.load(f)
                
            bird_queries = [item['question'] for item in bird_test]
            bird_gold_dbs = [item['db_id'] for item in bird_test]
            
            top_k_results = run_retrieval(model, bird_queries, bird_gold_dbs, schemas, k=20)
            
            results = []
            for q, g, r in zip(bird_queries, bird_gold_dbs, top_k_results):
                results.append({"question": q, "gold_db": g, "retrieved_dbs": r})
            
            with open(os.path.join(OUTPUT_DIR, "bird_retrieval_results.json"), 'w') as f:
                json.dump(results, f, indent=2)
                
            metrics = calculate_metrics("Bird", bird_queries, bird_gold_dbs, top_k_results)
            all_metrics.append(metrics)

        except Exception as e:
            print(f"Error processing Bird: {e}")

    if all_metrics:
        save_metrics_to_csv(all_metrics)

if __name__ == "__main__":
    main()
