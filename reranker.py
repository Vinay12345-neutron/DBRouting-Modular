"""
Local LLM–Based Re-ranking Script for Database Query Routing

This script performs re-ranking of candidate databases for natural language
queries in cross-domain NL-to-SQL query routing tasks using a locally hosted
large language model (LLM).

Overview:
- Takes initial database retrieval results (top-K candidates per query).
- Uses a local instruction-tuned LLM to:
  1. Generate table-level adjacency lists from database schemas.
  2. Map phrases in a natural language query to schema entities.
- Scores each candidate database using deterministic heuristics:
  - Coverage score: penalizes unmapped query phrases.
  - Connectivity score: checks whether referenced tables form a connected
    subgraph under foreign-key relationships.
  - Optional semantic similarity score using precomputed embeddings.
- Re-ranks candidate databases based on the combined score.

Key Characteristics:
- All LLM inference is performed locally (no external API calls).
- No model training or fine-tuning is performed.
- Designed for explainable, rule-based re-ranking rather than end-to-end
  learned ranking.

Input:
- Retrieval results in JSON format:
  `results/<dataset>_retrieval_results.json`
- Database schemas loaded via `load_schemas()`.

Output:
- Re-ranked results saved to:
  `results/<dataset>_reranked_local.json`
- Persistent cache of generated adjacency lists:
  `adjacency_lists_local.json`

Supported Datasets:
- Spider
- BIRD (BirdSQL)

Usage:
- Ensure processed datasets and retrieval results are available.
- Configure the local LLM model name if required.
- Run the script to generate re-ranked outputs for all configured datasets.

Notes:
- This script is intended for research experimentation and ablation studies
  in query routing, not for production deployment.
"""

import os
import json
import time
import torch
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm

# --- Configuration ---
DATA_DIR = "processed_data"
RESULTS_DIR = "results"
ADJACENCY_FILE = "adjacency_lists_local.json" # Distinct file for local run
# We process BOTH datasets now
DATASETS = ["spider", "bird"]

# Local LLM Config
LLM_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

class LocalLLM:
    def __init__(self, model_name: str):
        print(f"Loading Local LLM: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="auto", 
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        self.device = self.model.device
        print(f"Local LLM Loaded on {self.device}.")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                pad_token_id=self.tokenizer.eos_token_id,
                temperature=0.1,
                do_sample=False
            )
            
        # Decode only the NEW tokens
        generated_ids = output_ids[0][inputs.input_ids.shape[-1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        # Clean markdown code blocks if present
        generated_text = generated_text.replace("```json", "").replace("```", "").strip()
        return generated_text

# Initialize LLM Global (Lazy load in main or global?)
# Better to do inside main to avoid import side effects if imported elsewhere
llm_engine = None

# --- Prompts ---

def get_adjacency_prompt(schema_text: str) -> Tuple[str, str]:
    sys = (
        "You are a Senior Database Architect. "
        "You MUST follow the instructions exactly. "
        "Output ONLY valid JSON. "
        "No explanations, no markdown, no extra text."
    )

    usr = f"""
ROLE
You are a Senior Database Architect. Your expertise lies in analyzing relational database schemas to understand data flow, dependencies, and
potential join paths based on structure, naming conventions, and explicit constraints.

OBJECTIVE
Analyze the provided database schema and generate a directed adjacency list representing potential direct JOIN relationships between
tables (ONLY INNER JOINS). The output must strictly use numerical indices for tables and adhere precisely to the specified format, reflecting
both explicit and implied relationships suitable for join operations.

INPUT SCHEMA
[Schema Information]
{schema_text}

ANALYSIS AND PROCESSING STEPS
1. Table Indexing: Assign a unique, sequential, 0-based numerical index to each table defined in the schema based on the order they appear
or are logically grouped.

2. Relationship Identification (Apply in order of priority): For each Table A (with index idx_A):
   Identify likely Primary Key(s) for all tables. Assume standard PK naming conventions (e.g., id, uuid, <table_name>_id,
   <table_name>_uuid, <table_name>_code) and PK constraints if specified.

   Priority 1: Explicit Foreign Keys (FKs):
   If Table A has an FK column that references the PK of Table B (with index idx_B), establish a directed relationship idx_A → idx_B.

   Priority 2: Naming Conventions & Data Type Matching (Implied FKs):
   If a column in Table A follows common FK naming patterns and matches a compatible PK in Table B, establish idx_A → idx_B.

   Priority 3: Simple Name/Type Match (Lower Confidence):
   If a column in Table A has the same data type as the likely PK of Table B and the name implies a relationship, establish idx_A → idx_B.

3. Reciprocity Handling (VERY IMPORTANT):
   If a relationship idx_A → idx_B is established, also establish idx_B → idx_A to represent bidirectional JOIN capability.

4. Consolidation:
   For each source table index, collect all unique target table indices and sort them numerically.

OUTPUT SPECIFICATION
Format: The output MUST be a JSON object encoding the adjacency list.

Rules:
- Keys MUST be numerical table indices as strings ("0", "1", ..., "N-1")
- Values MUST be lists of numerical table indices
- Target indices MUST be sorted in ascending order
- Every table index from 0 to N-1 MUST be present
- Tables with no joins MUST have an empty list
- ONLY INNER JOIN relationships
- ABSOLUTELY NO table names, column names, explanations, comments, or extra text

Example:
{{
  "0": [1,2,5],
  "1": [0],
  "2": [],
  "3": [4],
  "4": [3],
  "5": [0]
}}
"""
    return sys, usr

def get_mapping_prompt(query: str, schema_text: str) -> Tuple[str, str]:
    sys = (
        "You are an expert Database Administrator specializing in NLP and Schema Mapping. "
        "Follow the instructions EXACTLY. "
        "Output ONLY plain text mappings in the specified format. "
        "No explanations, no markdown, no extra text."
    )

    usr = f"""
Role: You are an expert Database Administrator specializing in Natural Language Processing (NLP) and Schema Mapping.

Objective:
Analyze a given natural language QUERY and map meaningful words or phrases within it to the most relevant columns in
the provided database TABLE INFORMATION. Your goal is to identify the semantic connection between the user’s query terms
and the underlying data structure.

Inputs:
1. QUERY: The natural language question posed by the user.
2. TABLE INFORMATION: A description of the database schema, including table names and column definitions.

FOLLOW THESE INSTRUCTIONS VERY STRICTLY:

1. Analyze Context:
Thoroughly examine both the QUERY and the TABLE INFORMATION. Understand relationships between tables (via Foreign Keys if provided)
and the likely data stored in each column.

2. Identify Mappable Terms:
Extract words or multi-word phrases that carry semantic weight.
Focus on:
- Nouns (entities/attributes)
- Verbs (actions implying status/entities)
- Adjectives/Adverbs ONLY if they represent actual column values or states
- Specific literal values (names, numbers, status values)

DO NOT MAP:
- SQL keywords (SELECT, WHERE, etc.)
- Aggregate phrases ("number of", "how many")
- General modifiers ("different", "unique", "lowest", etc.)
- Stop words ("what", "the", "in", etc.)

Special Rule:
- "number of students" → map ONLY "students" to identifier column
- "student number" → actual ID column

3. Perform Column Mapping:
- Direct & semantic matches
- Verb-to-concept mappings
- Entity-to-identifier mappings
- Multiple valid mappings allowed
- Use N/A ONLY if no justified mapping exists

4. Precision:
All mappings must be directly justifiable. No speculation.

OUTPUT FORMAT (STRICT):
Each mapping MUST be on its own line in this exact format:
word_or_phrase - TableName.ColumnName

If no match exists:
word_or_phrase - N/A

— START INPUT —
QUERY: {query}

TABLE INFORMATION:
{schema_text}
— END INPUT —

ABSOLUTELY NO OTHER OUTPUT.
"""
    return sys, usr

# --- Step 1: Adjacency List Generation ---

def generate_adjacency_list(db_id: str, schema_text: str, cache: Dict = {}) -> Dict[str, List[str]]:
    if db_id in cache: return cache[db_id]
    
    # Check persistent cache
    if os.path.exists(ADJACENCY_FILE):
        try:
            with open(ADJACENCY_FILE, 'r') as f:
                file_data = json.load(f)
                if db_id in file_data:
                    cache[db_id] = file_data[db_id]
                    return file_data[db_id]
        except: pass

    # Generate via Local LLM
    sys, usr = get_adjacency_prompt(schema_text)
    global llm_engine
    
    try:
        text = llm_engine.generate(sys, usr)
        # parser
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end != -1:
            json_str = text[start:end]
            adj_list = json.loads(json_str)
        else:
            print(f"Failed to parse adjacency JSON for {db_id}. Raw: {text[:50]}...")
            adj_list = {}

        # Save to file
        if os.path.exists(ADJACENCY_FILE):
             with open(ADJACENCY_FILE, 'r') as f:
                try: file_data = json.load(f)
                except: file_data = {}
        else: file_data = {}
            
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
        # parser
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end != -1:
            json_str = text[start:end]
            data = json.loads(json_str)
            return data.get("mappings", [])
        return []
    except Exception as e:
        # print(f"Error mapping query: {e}")
        return []

# --- Step 3: Scoring Logic (Same as before) ---

def check_connectivity(mappings: List[Dict], adjacency: Dict[str, List[str]]) -> float:
    tables = set()
    for m in mappings:
        for entity in m['entities']:
            if "." in entity:
                parts = entity.split(".")
                tables.add(parts[0]) 
            else:
                tables.add(entity)

    if not tables: return 0.0
    if len(tables) == 1: return 1.0
    
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
                
    if tables.issubset(visited): return 1.0
    return 0.0

def calculate_score(query: str, db_id: str, schema_text: str, adjacency: Dict, embedding_model=None) -> Tuple[float, float]:
    mappings = map_query_to_schema(query, schema_text)
    
    total_phrases = len(mappings)
    if total_phrases == 0: return 0.0, 0.0
    
    na_mappings = sum(1 for m in mappings if not m['entities'] or m['entities'] == ["N/A"])
    x = na_mappings / total_phrases
    n = 2 
    coverage_score = np.exp(-n * x)
    
    connectivity_score = check_connectivity(mappings, adjacency)
    total_score = coverage_score * connectivity_score
    
    semantic_score = 0.0
    if embedding_model and mappings:
        sims = []
        for m in mappings:
            phrase = m['phrase']
            entities = m['entities'] 
            if not entities or entities == ["N/A"]: continue
            
            phrase_emb = embedding_model.encode([phrase], batch_size=1) 
            entity_embs = embedding_model.encode(entities, batch_size=len(entities)) 
            
            scores = torch.mm(torch.tensor(phrase_emb), torch.tensor(entity_embs).t()) 
            sims.append(scores.max().item())
            
        if sims: semantic_score = np.mean(sims)
            
    return total_score, semantic_score

# --- Main Re-ranking Loop ---

MAX_SAMPLES = None # Unlimited for Local LLM run

def process_dataset(dataset_name, schemas, embed_model, adj_cache):
    input_file = os.path.join(RESULTS_DIR, f"{dataset_name}_retrieval_results.json")
    output_file = os.path.join(RESULTS_DIR, f"{dataset_name}_reranked_local.json") 
    
    if not os.path.exists(input_file):
        print(f"Skipping {dataset_name}: {input_file} not found.")
        return

    print(f"\n=== Re-ranking {dataset_name} (Local LLM) ===")
    with open(input_file, 'r') as f:
        results = json.load(f)
        
    reranked_results = []
    
    # Process ALL or subset
    subset = results[:MAX_SAMPLES] if MAX_SAMPLES else results
    print(f"Processing {len(subset)} queries...")
    
    # Use tqdm for progress bar
    for i, item in enumerate(tqdm(subset, desc=f"Reranking {dataset_name}", unit="query")):
        query = item['question']
        candidates = item['retrieved_dbs']
        gold_db = item['gold_db']
        
        candidate_scores = []
        for db_id in candidates:
            schema = schemas.get(db_id, "")
            adj = generate_adjacency_list(db_id, schema, adj_cache)
            score, sem_score = calculate_score(query, db_id, schema, adj, embed_model)
            candidate_scores.append( (db_id, score, sem_score) )
            
        candidate_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
        
        reranked_results.append({
            "question": query,
            "gold_db": gold_db,
            "original_top1": candidates[0],
            "reranked_top1": candidate_scores[0][0],
            "reranked_candidates": candidate_scores
        })
        
        # Periodic Save every 20 queries
        if len(reranked_results) % 20 == 0:
             with open(output_file, 'w') as f:
                json.dump(reranked_results, f, indent=2)
                
    with open(output_file, 'w') as f:
        json.dump(reranked_results, f, indent=2)
    print(f"Done. Saved to {output_file}")

def main():
    global llm_engine
    llm_engine = LocalLLM(LLM_MODEL_NAME)

    # Load Embedding Model for Semantic Scoring
    print("Loading Embedding Model for Semantic Scoring...")
    try:
        from retrieval import EmbeddingModel
        embed_model = EmbeddingModel() 
    except Exception as e:
        print(f"Embedding model check failed: {e}. Scoring will rely on Coverage/Connectivity only.")
        embed_model = None
    
    from retrieval import load_schemas
    schemas = load_schemas() 
    adj_cache = {}
    
    for ds in DATASETS:
        process_dataset(ds, schemas, embed_model, adj_cache)

if __name__ == "__main__":
    main()
