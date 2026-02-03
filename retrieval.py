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
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
OUTPUT_DIR = "results"
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
        os.path.join(RAW_DATA_DIR, "spider_data", "tables.json"),
        os.path.join(RAW_DATA_DIR, "train", "train_tables.json"),
        # Add Bird paths if different
    ]
    
    print("Loading schemas...")
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

def main():
    # Initialize Model (Expects CUDA)
    try:
        model = EmbeddingModel()
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return

    schemas = load_schemas()
    if not schemas:
        print("No schemas loaded. Check data/ path.")
        return

    # Process Spider-Route
    spider_path = os.path.join(DATA_DIR, "spider_route_test.json")
    if os.path.exists(spider_path):
        print("\n=== Processing Spider-Route ===")
        try:
            with open(spider_path, 'r') as f:
                spider_test = json.load(f)
                
            spider_queries = [item['question'] for item in spider_test]
            spider_gold_dbs = [item['db_id'] for item in spider_test]
            
            # Run Retrieval
            top_k_results = run_retrieval(model, spider_queries, spider_gold_dbs, schemas)
            
            # Structuring Results
            results = []
            recall_1_count = 0
            for i, (query, gold, retrieved) in enumerate(zip(spider_queries, spider_gold_dbs, top_k_results)):
                results.append({
                    "question": query,
                    "gold_db": gold,
                    "retrieved_dbs": retrieved
                })
                if gold == retrieved[0]:
                    recall_1_count += 1
                
            # Metric Preview
            print(f"Spider Recall@1: {recall_1_count / len(results):.4f}")
            
            # Save Output
            out_file = os.path.join(OUTPUT_DIR, "spider_retrieval_results.json")
            with open(out_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Saved results to {out_file}")
            
        except Exception as e:
            print(f"Error processing Spider: {e}")
    else:
        print(f"Skipping Spider: {spider_path} not found.")

    # Process Bird-Route
    bird_path = os.path.join(DATA_DIR, "bird_route_test.json")
    if os.path.exists(bird_path):
        print("\n=== Processing Bird-Route ===")
        try:
            with open(bird_path, 'r') as f:
                bird_test = json.load(f)
                
            bird_queries = [item['question'] for item in bird_test]
            bird_gold_dbs = [item['db_id'] for item in bird_test]
            
            top_k_results = run_retrieval(model, bird_queries, bird_gold_dbs, schemas)
            
            results = []
            recall_1_count = 0
            for i, (query, gold, retrieved) in enumerate(zip(bird_queries, bird_gold_dbs, top_k_results)):
                results.append({
                    "question": query,
                    "gold_db": gold,
                    "retrieved_dbs": retrieved
                })
                if gold == retrieved[0]:
                    recall_1_count += 1
                    
            print(f"Bird Recall@1: {recall_1_count / len(results):.4f}")
            
            out_file = os.path.join(OUTPUT_DIR, "bird_retrieval_results.json")
            with open(out_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Saved results to {out_file}")

        except Exception as e:
            print(f"Error processing Bird: {e}")
    else:
        print(f"Skipping Bird: {bird_path} not found.")

if __name__ == "__main__":
    main()
