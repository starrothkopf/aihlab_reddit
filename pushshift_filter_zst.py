import json
import re
import hashlib
import logging
from datetime import datetime
from collections import defaultdict

INPUT_FILE = "combined_corpus_cleaned.ndjson"
OUTPUT_FILE = "combined_corpus_cleaned_deduplicated.ndjson"
DUPLICATES_FILE = "duplicates_for_inspection.ndjson"

log = logging.getLogger("reddit_deduper")
log.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
log.addHandler(handler)

GPT35_RELEASE = datetime(2022, 11, 30)
GPT4_RELEASE = datetime(2023, 3, 14)
GPT4O_RELEASE = datetime(2024, 5, 13)
GPT4O_REPLACES_GPT4 = datetime(2025, 4, 30)
GPT5_RELEASE = datetime(2025, 8, 7)

def infer_models_from_date(created_date):
    created = datetime.strptime(created_date, "%Y-%m-%d %H:%M:%S")
    
    if created < GPT35_RELEASE:
        return []  # before ChatGPT release
    elif created < GPT4_RELEASE:
        return ["gpt-3.5"]  # only 3.5 available
    elif created < GPT4O_RELEASE:
        return ["gpt-4"]  # only 4 available
    elif created < GPT4O_REPLACES_GPT4:
        return ["gpt-4", "gpt-4o"]  # both 4 and 4o available
    elif created < GPT5_RELEASE:
        return ["gpt-4o"]  # 4o has replaced 4
    else:
        return ["gpt-5"]  # GPT-5 era

def get_text_content(obj):
    if obj.get("type") == "submission":
        title = obj.get("title", "").strip()
        selftext = obj.get("selftext", "").strip()
        return f"{title} {selftext}".strip()
    else:
        return obj.get("body", "").strip()

def deduplicate_corpus(input_file, output_file, duplicates_file):
    log.info(f"reading from: {input_file}")
    
    # dictionary: reddit_id -> merged entry
    unique_entries = {}
    
    # track all entries for each ID (including duplicates)
    all_entries_by_hash = defaultdict(list)
    
    total_lines = 0
    entries_before_cutoff = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            total_lines += 1
            
            try:
                entry = json.loads(line.strip())
                
                # skip entries before GPT-3.5 release
                created_date = entry.get("created_date")
                if created_date:
                    created = datetime.strptime(created_date, "%Y-%m-%d %H:%M:%S")
                    if created < GPT35_RELEASE:
                        entries_before_cutoff += 1
                        continue
                
                # get Reddit ID for deduplication
                reddit_id = entry.get("id")
                if not reddit_id:
                    continue
                
                # store this entry in the list for this ID
                all_entries_by_hash[reddit_id].append(entry)
                
                # if this ID already exists, merge model detections
                if reddit_id in unique_entries:
                    existing = unique_entries[reddit_id]
                    
                    # ensure model_detected is a list
                    if not isinstance(existing["model_detected"], list):
                        existing["model_detected"] = [existing["model_detected"]]
                    
                    # add new model if not already present
                    new_model = entry.get("model_detected")
                    if new_model and new_model not in existing["model_detected"]:
                        existing["model_detected"].append(new_model)
                        
                else:
                    # first time seeing this ID
                    model = entry.get("model_detected")
                    entry["model_detected"] = [model] if model else []
                    unique_entries[reddit_id] = entry
                
                if total_lines % 100000 == 0:
                    log.info(f"processed {total_lines:,} lines, unique so far: {len(unique_entries):,}")
                    
            except (json.JSONDecodeError, ValueError) as e:
                continue
    
    log.info(f"\ntotal lines read: {total_lines:,}")
    log.info(f"entries before GPT-3.5 release (filtered): {entries_before_cutoff:,}")
    log.info(f"unique entries: {len(unique_entries):,}")
    log.info(f"duplicates removed: {total_lines - entries_before_cutoff - len(unique_entries):,}")
    
    # write duplicates to separate file
    log.info(f"\nwriting duplicates to: {duplicates_file}")
    duplicate_groups = 0
    total_duplicates_written = 0
    
    with open(duplicates_file, 'w', encoding='utf-8') as dup_file:
        for reddit_id, entries_list in all_entries_by_hash.items():
            if len(entries_list) > 1:
                # this ID has duplicates
                duplicate_groups += 1
                
                # create a group entry showing all duplicates
                duplicate_group = {
                    "reddit_id": reddit_id,
                    "duplicate_count": len(entries_list),
                    "text_preview": get_text_content(entries_list[0])[:200],
                    "created_date": entries_list[0].get("created_date"),
                    "entries": entries_list
                }
                
                dup_file.write(json.dumps(duplicate_group) + '\n')
                total_duplicates_written += len(entries_list) - 1  # don't count the first one
    
    log.info(f"duplicate groups found: {duplicate_groups:,}")
    log.info(f"total duplicate entries: {total_duplicates_written:,}")
    
    # add temporal inference and write unique entries
    log.info(f"\nadding temporal model inference...")
    
    model_stats = defaultdict(int)
    inferred_stats = defaultdict(int)
    multiple_models_count = 0
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for reddit_id, entry in unique_entries.items():
            # add inferred models based on creation date
            created_date = entry.get("created_date")
            if created_date:
                inferred_models = infer_models_from_date(created_date)
                entry["models_inferred_temporal"] = inferred_models
                
                for model in inferred_models:
                    inferred_stats[model] += 1
            else:
                entry["models_inferred_temporal"] = []
            
            # sort model_detected list for consistency
            if entry["model_detected"]:
                entry["model_detected"] = sorted(entry["model_detected"])
                
                # count entries with multiple models
                if len(entry["model_detected"]) > 1:
                    multiple_models_count += 1
                
                # stats for detected models
                for model in entry["model_detected"]:
                    model_stats[model] += 1
            
            f.write(json.dumps(entry) + '\n')
    
    log.info(f"\noutput written to: {output_file}")
    
    log.info("\nexplicit model counts:")
    for model, count in sorted(model_stats.items()):
        log.info(f"  {model}: {count:,}")
    
    log.info("\ninferred model counts (temporal):")
    for model, count in sorted(inferred_stats.items()):
        log.info(f"  {model}: {count:,}")
    
    log.info(f"\nentries mentioning multiple models: {multiple_models_count:,}")

if __name__ == "__main__":
    deduplicate_corpus(INPUT_FILE, OUTPUT_FILE, DUPLICATES_FILE)