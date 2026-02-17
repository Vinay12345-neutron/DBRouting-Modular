"""
Diagnostic script to analyze schema coverage for Spider retrieval
"""
import json
from collections import Counter

print("=" * 80)
print("SCHEMA COVERAGE ANALYSIS")
print("=" * 80)

# Load all schema files
spider_tables = json.load(open('processed_data/spider_tables.json'))
spider_test_tables = json.load(open('processed_data/spider_test_tables.json'))
dev_tables = json.load(open('processed_data/dev_tables.json'))
train_tables = json.load(open('processed_data/train_tables.json'))

# Load test queries
spider_test_queries = json.load(open('processed_data/spider_route_test.json'))
bird_test_queries = json.load(open('processed_data/bird_route_test.json'))

# Extract DB IDs
spider_tables_dbs = set(x['db_id'] for x in spider_tables)
spider_test_tables_dbs = set(x['db_id'] for x in spider_test_tables)
dev_tables_dbs = set(x['db_id'] for x in dev_tables)
train_tables_dbs = set(x['db_id'] for x in train_tables)

spider_gold_dbs = set(x['db_id'] for x in spider_test_queries)
bird_gold_dbs = set(x['db_id'] for x in bird_test_queries)

print("\n1. SCHEMA FILE COVERAGE:")
print(f"   spider_tables.json:      {len(spider_tables_dbs):>3} DBs")
print(f"   spider_test_tables.json: {len(spider_test_tables_dbs):>3} DBs")
print(f"   dev_tables.json (Bird):  {len(dev_tables_dbs):>3} DBs")
print(f"   train_tables.json (Bird):{len(train_tables_dbs):>3} DBs")

print("\n2. TEST QUERY REQUIREMENTS:")
print(f"   Spider test needs: {len(spider_gold_dbs):>3} unique DBs (from {len(spider_test_queries)} queries)")
print(f"   Bird test needs:   {len(bird_gold_dbs):>3} unique DBs (from {len(bird_test_queries)} queries)")

# Check coverage
all_spider_schemas = spider_tables_dbs | spider_test_tables_dbs
missing_spider = spider_gold_dbs - all_spider_schemas
missing_spider_in_bird_dev = missing_spider & dev_tables_dbs
missing_spider_in_bird_train = missing_spider & train_tables_dbs
truly_missing = missing_spider - dev_tables_dbs - train_tables_dbs

print("\n3. SPIDER SCHEMA GAPS:")
print(f"   DBs in spider_tables.json:      {len(spider_tables_dbs):>3}")
print(f"   DBs in spider_test_tables.json: {len(spider_test_tables_dbs):>3}")
print(f"   Union of both files:            {len(all_spider_schemas):>3}")
print(f"   Missing from Spider files:      {len(missing_spider):>3}")
print(f"     Found in dev_tables (Bird):   {len(missing_spider_in_bird_dev):>3} - {sorted(missing_spider_in_bird_dev)}")
print(f"     Found in train_tables (Bird): {len(missing_spider_in_bird_train):>3} - {sorted(missing_spider_in_bird_train)}")
print(f"     TRULY MISSING (not anywhere):  {len(truly_missing):>3}")

if truly_missing:
    print(f"\n   ⚠️  CRITICAL: {len(truly_missing)} schemas not found anywhere:")
    for db in sorted(truly_missing):
        print(f"       - {db}")

# Calculate impact
affected_queries = [q for q in spider_test_queries if q['db_id'] in missing_spider]
print(f"\n4. IMPACT:")
print(f"   Queries affected by missing schemas: {len(affected_queries)}/{len(spider_test_queries)} ({100*len(affected_queries)/len(spider_test_queries):.1f}%)")

# Distribution
db_freq = Counter(q['db_id'] for q in spider_test_queries)
print(f"\n5. MOST QUERIED DBs IN SPIDER TEST:")
for db, count in db_freq.most_common(10):
    status = "✓" if db in all_spider_schemas else "✗ MISSING"
    print(f"   {db:30s} {count:>4} queries  {status}")

print("\n6. CURRENT retrieval.py LOADING LOGIC:")
print("   Files checked by load_schemas():")
print("     - processed_data/spider_table.json")
print("     - processed_data/tables.json") 
print("     - data/spider_data/tables.json")
print("     - processed_data/dev_tables.json")
print("     - processed_data/train_tables.json")
print("     - data/train/train_tables.json")
print("\n   ⚠️  NOTE: 'spider_table.json' (singular) vs 'spider_tables.json' (plural)")
print("   ⚠️  NOTE: Missing 'spider_test_tables.json' in search paths!")

print("\n" + "=" * 80)
print("RECOMMENDATION:")
print("=" * 80)
if missing_spider_in_bird_dev or missing_spider_in_bird_train:
    print("✓ Good news: Missing Spider schemas ARE in Bird files")
    print("  → Fix: load_schemas() already loads dev_tables.json & train_tables.json")
    print("  → This should work IF those files are being loaded properly")
print("\n✓ Main issue: Need to add spider_test_tables.json to search paths")
print("✓ Also check: Is 'spider_table.json' (singular) even a real file?")
print("=" * 80)
