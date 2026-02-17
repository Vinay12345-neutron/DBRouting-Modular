import os
import json
import gc
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configuration ---
DATA_DIR = "processed_data"
RESULTS_DIR = "results"
ADJACENCY_FILE = "adjacency_lists_local.json"
DATASETS = ["spider", "bird"]

# LLM Config - Optimized for RTX 5090 / 4090
# 14B is too slow (20s/query). Switching to 7B for 3x speedup.
LLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# MODE: "local" or "api"
# "local": Uses local GPU (RTX 5090). Fast, Free, Private.
# "api": Uses DeepSeek API. Requires API key. No VRAM usage.
EXECUTION_MODE = "api" 

# Local settings
USE_8BIT_QUANTIZATION = False

# API settings
API_BASE_URL = "https://api.deepseek.com"
API_MODEL_NAME = "deepseek-chat"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Process ALL queries
MAX_SAMPLES = None 

class LLMEngine:
    def __init__(self, mode="local", model_name=LLM_MODEL_NAME, use_8bit=False, api_key=None):
        self.mode = mode
        print(f"Initializing LLM Engine in [{mode.upper()}] mode...")
        
        if mode == "local":
            print(f"Loading Local LLM: {model_name}...")
            print(f"8-bit quantization: {use_8bit}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            load_kwargs = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            
            if use_8bit and torch.cuda.is_available():
                print("Using 8-bit quantization (bitsandbytes)...")
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                )
                load_kwargs["quantization_config"] = bnb_config
                load_kwargs["device_map"] = "auto"
            else:
                load_kwargs["torch_dtype"] = torch.float16
                load_kwargs["device_map"] = "auto"
            
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
            self.device = self.model.device
            print(f"Local LLM Loaded on {self.device}.")
            
        elif mode == "api":
            try:
                from openai import OpenAI
                # Use provided key or env var
                key = api_key or os.getenv("DEEPSEEK_API_KEY")
                
                # Fallback: Manually read .env if os.getenv failed
                if not key and os.path.exists(".env"):
                    try:
                        with open(".env", "r") as f:
                            for line in f:
                                if line.strip().startswith("DEEPSEEK_API_KEY"):
                                    parts = line.split("=", 1)
                                    if len(parts) == 2:
                                        key = parts[1].strip().strip('"').strip("'")
                                        print("DEBUG: Loaded key manually from .env")
                                        break
                    except Exception as e:
                        print(f"DEBUG: Failed to read .env manually: {e}")

                if not key:
                    print(f"DEBUG: Current Directory: {os.getcwd()}")
                    print(f"DEBUG: .env file exists: {os.path.exists('.env')}")
                    # Print sanitized content of .env to see what is going on
                    if os.path.exists(".env"):
                        print("DEBUG: .env content:")
                        with open(".env", "r") as f:
                            for line in f:
                                print(f"  {line.strip()}")
                    
                    raise ValueError("DEEPSEEK_API_KEY not configured. Please check .env file.")
                
                self.client = OpenAI(api_key=key, base_url=API_BASE_URL)
                print(f"Connected to DeepSeek API ({API_MODEL_NAME})")
            except ImportError:
                print("Error: 'openai' package not installed. Run 'pip install openai'")
                raise

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if self.mode == "local":
            return self._generate_local(system_prompt, user_prompt)
        else:
            return self._generate_api(system_prompt, user_prompt)

    def _generate_api(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=API_MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=1024,
                temperature=0.1,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API Error: {e}")
            return "{}"

    def _generate_local(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception as e:
            print(f"Chat template error: {e}. Using fallback.")
            prompt = f"{system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=4096 
        ).to(self.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                pad_token_id=self.tokenizer.eos_token_id,
                temperature=0.1,
                do_sample=False,
                num_beams=1 
            )
        
        generated_ids = output_ids[0][inputs.input_ids.shape[-1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        generated_text = generated_text.replace("```json", "").replace("```", "").strip()
        
        del inputs, output_ids, generated_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return generated_text

# Global LLM instance
llm_engine = None

# --- Prompts ---

def get_adjacency_prompt(schema_text: str) -> Tuple[str, str]:
    sys = "You are an expert DB Administrator. Output ONLY valid JSON."
    usr = f"""Given the following database schema, produce an adjacency list representing possible joins between tables under Foreign Key relationships.

Input Schema:
{schema_text}

Output ONLY a JSON object where keys are Table Names and values are lists of Table Names they can join with.
Example: {{"TableA": ["TableB"], "TableB": ["TableA", "TableC"]}}
"""
    return sys, usr

def get_mapping_prompt(query: str, schema_text: str) -> Tuple[str, str]:
    sys = "You are an expert database assistant. Output ONLY valid JSON."
    usr = f"""Question: "{query}"

Schema:
{schema_text}

Task: Extract phrases from the question and map them to the corresponding Table.Column in the schema.
Identified phrases should include nouns, verbs, or adjectives that map to schema entities.
If a phrase maps to multiple columns or tables, list all of them.

Output ONLY a JSON object with this format:
{{
  "mappings": [
    {{ "phrase": "exact phrase from text", "entities": ["Table.Column", "Table.Column"] }},
    ...
  ]
}}
"""
    return sys, usr

# --- Step 1: Adjacency List Generation ---

def generate_adjacency_list(db_id: str, schema_text: str, cache: Dict = {}) -> Dict[str, List[str]]:
    if db_id in cache:
        return cache[db_id]
    
    # Check persistent cache
    if os.path.exists(ADJACENCY_FILE):
        try:
            with open(ADJACENCY_FILE, 'r') as f:
                file_data = json.load(f)
                if db_id in file_data:
                    cache[db_id] = file_data[db_id]
                    return file_data[db_id]
        except:
            pass

    # Generate via LLM
    sys, usr = get_adjacency_prompt(schema_text)
    global llm_engine
    
    try:
        text = llm_engine.generate(sys, usr)
        
        # Robust JSON extraction
        import re
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            adj_list = json.loads(json_str)
        else:
            print(f"Failed to parse adjacency JSON for {db_id}. Raw: {text[:100]}...")
            adj_list = {}

        # Save to file
        if os.path.exists(ADJACENCY_FILE):
            with open(ADJACENCY_FILE, 'r') as f:
                try:
                    file_data = json.load(f)
                except:
                    file_data = {}
        else:
            file_data = {}
        
        file_data[db_id] = adj_list
        with open(ADJACENCY_FILE, 'w') as f:
            json.dump(file_data, f, indent=2)
        
        cache[db_id] = adj_list
        return adj_list
        
    except Exception as e:
        print(f"Error generating adjacency for {db_id}: {e}")
        return {}

# --- Step 2: Query-Schema Mapping ---

def map_query_to_schema(query: str, schema_text: str) -> List[Dict]:
    sys, usr = get_mapping_prompt(query, schema_text)
    global llm_engine
    
    try:
        text = llm_engine.generate(sys, usr)
        
        # Robust JSON extraction
        import re
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            data = json.loads(json_str)
            return data.get("mappings", [])
        return []
        
    except Exception as e:
        return []

# --- Step 3: Scoring Logic ---

def check_connectivity(mappings: List[Dict], adjacency: Dict[str, List[str]]) -> float:
    tables = set()
    for m in mappings:
        for entity in m.get('entities', []):
            if "." in entity:
                parts = entity.split(".")
                tables.add(parts[0])
            else:
                tables.add(entity)

    if not tables:
        return 0.0
    if len(tables) == 1:
        return 1.0
    
    # BFS connectivity check
    start_node = list(tables)[0]
    visited = {start_node}
    queue = [start_node]
    
    while queue:
        curr = queue.pop(0)
        neighbors = adjacency.get(curr, [])
        for n in neighbors:
            if n not in visited:
                visited.add(n)
                queue.append(n)
    
    return 1.0 if tables.issubset(visited) else 0.0

def calculate_score(
    query: str, 
    db_id: str, 
    schema_text: str, 
    adjacency: Dict, 
    embedding_model=None
) -> Tuple[float, float]:
    mappings = map_query_to_schema(query, schema_text)
    
    total_phrases = len(mappings)
    if total_phrases == 0:
        return 0.0, 0.0
    
    # Coverage score
    na_mappings = sum(
        1 for m in mappings 
        if not m.get('entities') or m.get('entities') == ["N/A"]
    )
    x = na_mappings / total_phrases
    n = 2
    coverage_score = np.exp(-n * x)
    
    # Connectivity score
    connectivity_score = check_connectivity(mappings, adjacency)
    total_score = coverage_score * connectivity_score
    
    # Semantic score (optional if embedding model available)
    semantic_score = 0.0
    if embedding_model and mappings:
        try:
            sims = []
            for m in mappings:
                phrase = m.get('phrase', '')
                entities = m.get('entities', [])
                if not entities or entities == ["N/A"] or not phrase:
                    continue
                
                phrase_emb = embedding_model.encode([phrase], batch_size=1)
                entity_embs = embedding_model.encode(entities, batch_size=len(entities))
                
                scores = torch.mm(
                    torch.tensor(phrase_emb), 
                    torch.tensor(entity_embs).t()
                )
                sims.append(scores.max().item())
                
                # Clear memory
                del phrase_emb, entity_embs, scores
            
            if sims:
                semantic_score = np.mean(sims)
        except Exception as e:
            print(f"Semantic scoring error: {e}")
            semantic_score = 0.0
    
    return total_score, semantic_score

# --- Main Re-ranking Loop ---

def process_dataset(dataset_name, schemas, embed_model, adj_cache):
    input_file = os.path.join(RESULTS_DIR, f"{dataset_name}_retrieval_results.json")
    output_file = os.path.join(RESULTS_DIR, f"{dataset_name}_reranked_local.json")
    
    if not os.path.exists(input_file):
        print(f"Skipping {dataset_name}: {input_file} not found.")
        return

    print(f"\n=== Re-ranking {dataset_name} (4GB VRAM Optimized) ===")
    with open(input_file, 'r') as f:
        results = json.load(f)
    
    reranked_results = []
    
    # Process subset or all
    subset = results[:MAX_SAMPLES] if MAX_SAMPLES else results
    print(f"Processing {len(subset)} queries...")
    
    for i, item in enumerate(tqdm(subset, desc=f"Reranking {dataset_name}", unit="query")):
        query = item['question']
        candidates = item['retrieved_dbs']
        gold_db = item['gold_db']
        
        candidate_scores = []
        for db_id in candidates:
            schema = schemas.get(db_id, "")
            adj = generate_adjacency_list(db_id, schema, adj_cache)
            score, sem_score = calculate_score(query, db_id, schema, adj, embed_model)
            candidate_scores.append((db_id, score, sem_score))
        
        # Sort by (total_score, semantic_score)
        candidate_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
        
        reranked_results.append({
            "question": query,
            "gold_db": gold_db,
            "original_top1": candidates[0],
            "reranked_top1": candidate_scores[0][0],
            "reranked_candidates": candidate_scores
        })
        
        # Periodic save + garbage collection
        if (i + 1) % 20 == 0:
            with open(output_file, 'w') as f:
                json.dump(reranked_results, f, indent=2)
            
            # Aggressive memory cleanup for 4GB VRAM
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Final save
    with open(output_file, 'w') as f:
        json.dump(reranked_results, f, indent=2)
    print(f"Done. Saved to {output_file}")

def main():
    global llm_engine
    
    print("=" * 60)
    print("4GB VRAM Optimized Re-ranker")
    print("=" * 60)
    
    # Load LLM Engine
    llm_engine = LLMEngine(
        mode=EXECUTION_MODE, 
        model_name=LLM_MODEL_NAME, 
        use_8bit=USE_8BIT_QUANTIZATION,
        api_key=DEEPSEEK_API_KEY
    )
    
    # Load embedding model (optional for semantic scoring)
    print("\nLoading Embedding Model for Semantic Scoring...")
    embed_model = None
    try:
        from retrieval import EmbeddingModel
        embed_model = EmbeddingModel()
        print("Embedding model loaded successfully.")
    except Exception as e:
        print(f"Embedding model not available: {e}")
        print("Will skip semantic scoring (only coverage + connectivity).")
    
    # Load schemas
    from retrieval import load_schemas
    schemas = load_schemas()
    adj_cache = {}
    
    print(f"\nLoaded {len(schemas)} database schemas.")
    print(f"Max samples per dataset: {MAX_SAMPLES if MAX_SAMPLES else 'ALL'}")
    
    # Process datasets
    for ds in DATASETS:
        process_dataset(ds, schemas, embed_model, adj_cache)
        
        # Clear memory between datasets
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print("\n" + "=" * 60)
    print("Re-ranking Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
