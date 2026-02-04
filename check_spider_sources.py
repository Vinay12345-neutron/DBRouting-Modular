"""
Quick script to check original Spider data sources
"""
import os
import json

print("=" * 60)
print("CHECKING ORIGINAL SPIDER DATA SOURCES")
print("=" * 60)

# Search for Spider source files
source_locations = [
    "data",
    "spider",
    "Spider",
    ".",
]

spider_files = [
    "train_spider.json",
    "train_others.json", 
    "dev.json",
    "tables.json"
]

print("\n[1] Searching for Spider source files...")
for loc in source_locations:
    if not os.path.exists(loc):
        continue
    
    print(f"\nChecking: {loc}/")
    for root, dirs, files in os.walk(loc):
        for sf in spider_files:
            if sf in files:
                full_path = os.path.join(root, sf)
                size_mb = os.path.getsize(full_path) / (1024*1024)
                print(f"  ✓ Found: {full_path} ({size_mb:.1f} MB)")
                
                # Count DBs in tables.json if found
                if sf == "tables.json":
                    try:
                        with open(full_path, 'r') as f:
                            tables = json.load(f)
                            db_ids = set(db['db_id'] for db in tables)
                            print(f"    → Contains {len(db_ids)} unique databases")
                    except:
                        pass

print("\n[2] Checking processed data...")
if os.path.exists("processed_data/spider_route_test.json"):
    with open("processed_data/spider_route_test.json") as f:
        test_data = json.load(f)
    with open("processed_data/spider_route_train.json") as f:
        train_data = json.load(f)
    
    all_dbs = set(item['db_id'] for item in test_data + train_data)
    print(f"  Processed Spider-Route: {len(all_dbs)} DBs")
    print(f"  Expected (per paper): 206 DBs")
    
    if len(all_dbs) < 206:
        print(f"\n  ⚠ GAP: Missing {206 - len(all_dbs)} databases")
        print("  Possible reasons:")
        print("    1. Source files incomplete")
        print("    2. prepare_data.py didn't find all files")
        print("    3. Some DBs filtered out during processing")
    else:
        print(f"\n  ✓ Complete dataset!")

print("\n" + "=" * 60)
