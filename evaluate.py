"""
Evaluation Script for Database Retrieval and Re-ranking

This script evaluates database retrieval and re-ranking outputs for
cross-domain NL-to-SQL query routing tasks (e.g., Spider and BIRD).

Given a JSON file containing model predictions per query, it computes:
- Recall@K (K = 1, 3, 5): Whether the gold database appears in the top-K candidates.
- Mean Average Precision (mAP): Computed as 1 / rank of the gold database
  for single-ground-truth scenarios.

The script supports two output formats:
1. Retrieval-only results (`retrieved_dbs`)
2. Re-ranked results (`reranked_candidates`)

Each results file is expected to contain a list of query-level entries,
where each entry specifies the gold database and an ordered list of
predicted candidate databases.

Usage:
- Place result JSON files in the `results/` directory.
- Run the script to evaluate baseline retrieval and re-ranked outputs
  for Spider and BIRD datasets.
"""

import json
import os
import numpy as np
from sklearn.metrics import average_precision_score

RESULTS_DIR = "results"

def calculate_metrics(results_file):
    if not os.path.exists(results_file):
        print(f"File not found: {results_file}")
        return

    print(f"Evaluating {results_file}...")
    with open(results_file, 'r') as f:
        data = json.load(f)

    # 1. Recall@K
    recall_1 = []
    recall_3 = []
    recall_5 = []
    
    for item in data:
        gold = item['gold_db']
        
        # Check if 'reranked_candidates' exists (from reranker)
        if 'reranked_candidates' in item:
            # List of (db_id, score, sem_score)
            ranked_dbs = [x[0] for x in item['reranked_candidates']]
        elif 'retrieved_dbs' in item:
            # List of db_ids from retrieval
            ranked_dbs = item['retrieved_dbs']
        else:
            continue
            
        # Top-K check
        recall_1.append(1 if gold in ranked_dbs[:1] else 0)
        recall_3.append(1 if gold in ranked_dbs[:3] else 0)
        recall_5.append(1 if gold in ranked_dbs[:5] else 0)

    print(f"Total Queries: {len(data)}")
    print(f"Recall@1: {np.mean(recall_1):.4f}")
    print(f"Recall@3: {np.mean(recall_3):.4f}")
    print(f"Recall@5: {np.mean(recall_5):.4f}")

    # mAP (Mean Average Precision)
    # For single ground truth, mAP = 1/Rank if found, else 0
    aps = []
    for item in data:
        gold = item['gold_db']
        if 'reranked_candidates' in item:
            ranked_dbs = [x[0] for x in item['reranked_candidates']]
        elif 'retrieved_dbs' in item:
            ranked_dbs = item['retrieved_dbs']
        else:
            continue
            
        if gold in ranked_dbs:
            rank = ranked_dbs.index(gold) + 1
            aps.append(1.0 / rank)
        else:
            aps.append(0.0)
            
    print(f"mAP: {np.mean(aps):.4f}")

if __name__ == "__main__":
    print("--- Spider Retrieval Baseline (B1/B3 Equivalent) ---")
    calculate_metrics(os.path.join(RESULTS_DIR, "spider_retrieval_results.json"))
    
    print("\n--- Spider Re-ranked (Ours) ---")
    calculate_metrics(os.path.join(RESULTS_DIR, "spider_reranked_local.json"))

    print("\n--- Bird Retrieval Baseline ---")
    calculate_metrics(os.path.join(RESULTS_DIR, "bird_retrieval_results.json"))

    print("\n--- Bird Re-ranked (Ours) ---")
    calculate_metrics(os.path.join(RESULTS_DIR, "bird_reranked_local.json"))
