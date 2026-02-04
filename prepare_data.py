"""
Dataset Preparation Script for Query Routing Benchmarks (Spider & BIRD)

This script prepares training and test splits for database-level
query routing experiments using the Spider and BIRD (BirdSQL) datasets.

Functionality:
- Loads Spider train and dev JSON files and merges them into a single pool.
- Groups queries by database ID and performs a deterministic 50/50 split
  per database to create balanced train and test sets (Spider-Route).
- Loads BirdSQL train and dev data using content-based heuristics to
  distinguish it from Spider data.
- Performs an identical per-database 50/50 deterministic split to produce
  Bird-Route train and test sets.

Key Characteristics:
- Splitting is deterministic (no random seed required).
- Ensures each database is represented in both train and test sets.
- Designed for cross-domain NL-to-SQL query routing evaluation rather than
  SQL generation.

Input:
- Raw Spider and BIRD JSON files placed under the `data/` directory.

Output:
- `processed_data/spider_route_train.json`
- `processed_data/spider_route_test.json`
- `processed_data/bird_route_train.json`
- `processed_data/bird_route_test.json`

Usage:
- Organize datasets under the `data/` directory.
- Run the script once to generate processed routing splits.
- The resulting JSON files can be directly consumed by retrieval or
  re-ranking models for evaluation.
"""

import os
import json
import sqlite3
import pandas as pd
from typing import List, Dict

# Config
DATA_DIR = "data"
SPIDER_DIR = os.path.join(DATA_DIR, "spider")
BIRD_DEV_DIR = os.path.join(DATA_DIR, "bird_dev")
BIRD_TRAIN_DIR = os.path.join(DATA_DIR, "bird_train")

OUTPUT_DIR = "processed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_spider():
    """Load and merge Spider Train and Dev sets."""
    print("Loading Spider data...")
    
    data = []
    
    # Try finding files
    possible_files = [
        os.path.join(DATA_DIR, "train_spider.json"),
        os.path.join(DATA_DIR, "train_others.json"),
        os.path.join(DATA_DIR, "dev.json")
    ]
    
    for f in possible_files:
        if os.path.exists(f):
            print(f"Loading {f}...")
            with open(f, 'r') as fd:
                data.extend(json.load(fd))
        else:
            # Fallback search if paths differ
            filename = os.path.basename(f)
            found = False
            for root, dirs, files in os.walk(DATA_DIR):
                if filename in files:
                    full_path = os.path.join(root, filename)
                    print(f"Loading {full_path}...")
                    with open(full_path, 'r') as fd:
                        data.extend(json.load(fd))
                    found = True
                    break
            if not found:
                print(f"Warning: {f} not found.")
            
    return data

def process_spider(all_data):
    """
    Split queries 50/50 per database.
    Spider-Route: 206 DBs.
    """
    if not all_data:
        print("No Spider data found.")
        return

    print(f"Processing {len(all_data)} Spider queries...")
    
    # Group by db_id
    db_groups = {}
    for item in all_data:
        db_id = item.get('db_id')
        if not db_id: continue
        
        if db_id not in db_groups:
            db_groups[db_id] = []
        db_groups[db_id].append(item)
        
    train_queries = []
    test_queries = []
    
    # sort db_ids for deterministic behavior
    for db_id in sorted(db_groups.keys()):
        items = db_groups[db_id]
        # Deterministic split 50/50
        mid = len(items) // 2
        train_queries.extend(items[:mid])
        test_queries.extend(items[mid:])
        
    print(f"Spider-Route: Train {len(train_queries)}, Test {len(test_queries)}")
    
    with open(os.path.join(OUTPUT_DIR, "spider_route_train.json"), 'w') as f:
        json.dump(train_queries, f, indent=2)
        
    with open(os.path.join(OUTPUT_DIR, "spider_route_test.json"), 'w') as f:
        json.dump(test_queries, f, indent=2)

def load_bird():
    """Load and merge BirdSQL Train and Dev sets."""
    print("Loading BIRD data...")
    
    data = []
    
    # Walk to find all dev.json and train.json that look like BirdSQL data
    # BirdSQL typically has 'question' and 'SQL' keys and 'evidence'
    for root, dirs, files in os.walk(DATA_DIR):
        for filename in ["dev.json", "train.json"]:
            if filename in files:
                f_path = os.path.join(root, filename)
                
                # Heuristic: Check size or content to distinguish from Spider
                try:
                    with open(f_path, 'r', encoding='utf-8') as f:
                        # Read first chunk to avoid OOM on huge files just for checking
                        # But standard json load is fine for <1GB usually. 
                        # BIRD train.json is large, might be better to check path structure.
                        
                        # Optimization: check path contains 'bird' if possible, or assume user unzipped structure
                        # If the user just unzipped everything to data/, filenames might clash if not in subfolders.
                        # Assuming the unzipping preserved subfolders.
                        
                        # Let's try loading.
                        content = json.load(f)
                        if isinstance(content, list) and len(content) > 0:
                            # Check keys of first item
                            keys = content[0].keys()
                            if 'evidence' in keys: # BirdSQL specific
                                print(f"Identified BIRD data: {f_path}")
                                data.extend(content)
                            else:
                                print(f"Skipping {f_path} (likely Spider data or other)")
                except Exception as e:
                    print(f"Error reading {f_path}: {e}")
                    
    return data

def process_bird(all_data):
    """
    Split queries 50/50 per database.
    Bird-Route: 80 DBs.
    """
    if not all_data:
        print("No BIRD data found.")
        return

    print(f"Processing {len(all_data)} BIRD queries...")
    
    db_groups = {}
    for item in all_data:
        db_id = item.get('db_id')
        if not db_id: continue

        if db_id not in db_groups:
            db_groups[db_id] = []
        db_groups[db_id].append(item)
        
    train_queries = []
    test_queries = []
    
    for db_id in sorted(db_groups.keys()):
        items = db_groups[db_id]
        mid = len(items) // 2
        train_queries.extend(items[:mid])
        test_queries.extend(items[mid:])
        
    print(f"Bird-Route: Train {len(train_queries)}, Test {len(test_queries)}")
    
    with open(os.path.join(OUTPUT_DIR, "bird_route_train.json"), 'w') as f:
        json.dump(train_queries, f, indent=2)
        
    with open(os.path.join(OUTPUT_DIR, "bird_route_test.json"), 'w') as f:
        json.dump(test_queries, f, indent=2)

if __name__ == "__main__":
    spider_data = load_spider()
    process_spider(spider_data)
    
    bird_data = load_bird()
    process_bird(bird_data)
        
    print("Data preparation complete.")
