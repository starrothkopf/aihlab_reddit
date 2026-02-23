import json
import logging
from datetime import datetime
from collections import defaultdict

# stored in models_inferred_temporal

INPUT_FILE = "combined_corpus_cleaned.ndjson"
OUTPUT_FILE = "combined_corpus_cleaned_inferred_models.ndjson"

log = logging.getLogger("infer_model")
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
        return []
    elif created < GPT4_RELEASE:
        return ["gpt-3.5"]
    elif created < GPT4O_RELEASE:
        return ["gpt-4"]
    elif created < GPT4O_REPLACES_GPT4:
        return ["gpt-4", "gpt-4o"]
    elif created < GPT5_RELEASE:
        return ["gpt-4o"]
    else:
        return ["gpt-5"]

def run_temporal_inference(input_file, output_file):
    log.info(f"reading from: {input_file}")

    total_lines = 0
    skipped_before_cutoff = 0
    inferred_stats = defaultdict(int)

    with open(input_file, "r", encoding="utf-8") as infile, \
         open(output_file, "w", encoding="utf-8") as outfile:

        for line in infile:
            total_lines += 1

            try:
                entry = json.loads(line.strip())
                created_date = entry.get("created_date")

                if created_date:
                    created = datetime.strptime(created_date, "%Y-%m-%d %H:%M:%S")
                    if created < GPT35_RELEASE:
                        skipped_before_cutoff += 1
                        continue

                    inferred_models = infer_models_from_date(created_date)
                    entry["models_inferred_temporal"] = inferred_models

                    for model in inferred_models:
                        inferred_stats[model] += 1
                else:
                    entry["models_inferred_temporal"] = []

                outfile.write(json.dumps(entry) + "\n")

                if total_lines % 100000 == 0:
                    log.info(f"processed {total_lines:,} lines")

            except (json.JSONDecodeError, ValueError):
                continue

    log.info(f"\ntotal lines read: {total_lines:,}")
    log.info(f"entries before GPT-3.5 release (filtered): {skipped_before_cutoff:,}")
    log.info(f"output written to: {output_file}")

    log.info("\ninferred model counts:")
    for model, count in sorted(inferred_stats.items()):
        log.info(f"  {model}: {count:,}")

if __name__ == "__main__":
    run_temporal_inference(INPUT_FILE, OUTPUT_FILE)