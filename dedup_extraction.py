import json
import hashlib
import logging
from collections import defaultdict

INPUT_FILE = "combined_corpus_cleaned.ndjson"
DUPLICATES_FILE = "duplicates_for_inspection.ndjson"
STATS_FILE = "duplicate_stats.txt"

log = logging.getLogger("duplicate_finder")
log.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
log.addHandler(handler)

def get_combined_text(obj):
    text_parts = []
    
    # get text based on type
    if obj.get("type") == "submission":
        if obj.get("title"):
            text_parts.append(obj["title"])
        if obj.get("selftext"):
            text_parts.append(obj["selftext"])
    else:  # comment
        if obj.get("body"):
            text_parts.append(obj["body"])
    
    return " ".join(text_parts).strip()


def find_duplicates(input_file, duplicates_file, stats_file):
    """
    find all duplicates with the exact same text content
    """
    
    log.info(f"Reading: {input_file}")
    
    # dictionary: text_hash -> list of full entries
    hash_to_entries = defaultdict(list)
    
    total = 0
    
    # first pass: group all entries by text hash
    with open(input_file, "r", encoding="utf-8") as infile:
        for line in infile:
            total += 1
            
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            # get combined text
            text = get_combined_text(obj)
            
            if not text:
                continue
            
            # create hash
            text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
            
            # store the full entry
            hash_to_entries[text_hash].append(obj)
            
            if total % 100000 == 0:
                log.info(f"Processed {total:,} lines...")
    
    log.info(f"total entries processed: {total:,}")
    log.info(f"unique text hashes: {len(hash_to_entries):,}")
    
    # find groups with duplicates (more than 1 entry)
    duplicate_groups = {k: v for k, v in hash_to_entries.items() if len(v) > 1}
    
    total_duplicate_entries = sum(len(v) for v in duplicate_groups.values())
    excess_duplicates = sum(len(v) - 1 for v in duplicate_groups.values())
    
    log.info(f"duplicate groups found: {len(duplicate_groups):,}")
    log.info(f"total entries in duplicate groups: {total_duplicate_entries:,}")
    log.info(f"excess duplicates (to be removed): {excess_duplicates:,}")
    
    # write all duplicate entries to file
    log.info(f"\nwriting duplicates to: {duplicates_file}")
    
    with open(duplicates_file, "w", encoding="utf-8") as outfile:
        for text_hash, entries in duplicate_groups.items():
            for entry in entries:
                # Add metadata about the duplicate group
                entry["_duplicate_group_hash"] = text_hash
                entry["_duplicate_group_size"] = len(entries)
                outfile.write(json.dumps(entry) + "\n")
    
    log.info(f"duplicates saved!")
    
    # Write detailed stats
    log.info(f"\nwriting stats to: {stats_file}")
    
    with open(stats_file, "w", encoding="utf-8") as statsfile:
        
        statsfile.write(f"total entries processed: {total:,}\n")
        statsfile.write(f"unique text hashes: {len(hash_to_entries):,}\n")
        statsfile.write(f"duplicate groups: {len(duplicate_groups):,}\n")
        statsfile.write(f"total entries in duplicate groups: {total_duplicate_entries:,}\n")
        statsfile.write(f"excess duplicates (to remove): {excess_duplicates:,}\n")
        statsfile.write(f"keep rate after dedup: {(total - excess_duplicates) / total * 100:.2f}%\n\n")
        
        # distribution of duplicate group sizes
        size_distribution = defaultdict(int)
        for entries in duplicate_groups.values():
            size_distribution[len(entries)] += 1
        
        statsfile.write("DUPLICATE GROUP SIZE DISTRIBUTION:\n")
        for size in sorted(size_distribution.keys()):
            count = size_distribution[size]
            statsfile.write(f"  {size} duplicates: {count:,} groups\n")
        
        statsfile.write("SAMPLE DUPLICATE GROUPS (first 20):\n")
        for i, (text_hash, entries) in enumerate(list(duplicate_groups.items())[:20]):
            statsfile.write(f"\nGroup {i+1} (hash: {text_hash[:16]}...):\n")
            statsfile.write(f"  Size: {len(entries)} entries\n")
            statsfile.write(f"  Type: {entries[0].get('type')}\n")
            statsfile.write(f"  Subreddit: {entries[0].get('subreddit')}\n")
            
            # get text preview
            text = get_combined_text(entries[0])
            preview = text[:150] + "..." if len(text) > 150 else text
            statsfile.write(f"  Text: {preview}\n")
            
            statsfile.write("  Entries:\n")
            for j, entry in enumerate(entries[:10]):  # show max 10 per group
                statsfile.write(f"    [{j+1}] ID: {entry.get('id')}, "
                              f"Model: {entry.get('model_detected')}, "
                              f"Date: {entry.get('created_date')}, "
                              f"Author: {entry.get('author')}\n")
            
            if len(entries) > 10:
                statsfile.write(f"    ... and {len(entries) - 10} more\n")
            
            statsfile.write("-" * 80 + "\n")
    
    log.info(f"total entries: {total:,}")
    log.info(f"unique entries: {total - excess_duplicates:,}")
    log.info(f"duplicate groups: {len(duplicate_groups):,}")
    log.info(f"excess duplicates to remove: {excess_duplicates:,}")
    log.info(f"\nfiles created:")
    log.info(f"  - {duplicates_file} (all duplicate entries)")
    log.info(f"  - {stats_file} (detailed analysis)")

if __name__ == "__main__":
    find_duplicates(INPUT_FILE, DUPLICATES_FILE, STATS_FILE)