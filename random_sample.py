import json
import random
from pathlib import Path

FOLDERS = [
    "pushshift_25_08_json",
    "pushshift_25_09_json",
    "pushshift_25_10_json",
    "pushshift_25_11_json",
    "pushshift_25_12_json",
    "pushshift_22_24_json",
]

TOTAL_SAMPLES = 300

all_entries = []

for folder in FOLDERS:
    folder_path = Path(folder)
    if not folder_path.exists():
        print(f"Skipping missing folder: {folder}")
        continue

    for json_file in folder_path.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        all_entries.append(entry)
                    except json.JSONDecodeError:
                        print(f"Bad JSON line in {json_file} at line {line_num}")
        except Exception as e:
            print(f"Failed to open {json_file}: {e}")

print(f"Total entries collected: {len(all_entries)}")

if len(all_entries) < TOTAL_SAMPLES:
    raise ValueError("Not enough entries to sample from.")

sampled_entries = random.sample(all_entries, TOTAL_SAMPLES)

output_path = Path("random_pushshift_sample.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(sampled_entries, f, indent=2, ensure_ascii=False)

print(f"Saved {TOTAL_SAMPLES} random entries to {output_path}")
