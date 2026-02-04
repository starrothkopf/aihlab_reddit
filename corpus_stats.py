#!/usr/bin/env python3
import json

CORPUS_FILE = "combined_corpus.ndjson"

total_lines = 0
total_chars = 0
total_body_chars = 0
types = {}
models_detected = {}
dates = {}

with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        
        total_lines += 1
        total_chars += len(line)
        
        try:
            data = json.loads(line)
            
            t = data.get('type', 'unknown')
            types[t] = types.get(t, 0) + 1
            
            body = data.get('body', '')
            total_body_chars += len(body)
            
            model = data.get('model_detected', 'none')
            models_detected[model] = models_detected.get(model, 0) + 1
            
            date = data.get('created_date', '')
            if date:
                year_month = date[:7]  # YYYY-MM
                dates[year_month] = dates.get(year_month, 0) + 1
                
        except json.JSONDecodeError:
            pass


print(f"\ntotal records: {total_lines:,}")
print(f"total file size (chars): {total_chars:,}")
print(f"total body text (chars): {total_body_chars:,}")
print(f"average body length: {total_body_chars/total_lines if total_lines > 0 else 0:.1f} chars")

print(f"\ntypes:")
for t, count in sorted(types.items(), key=lambda x: x[1], reverse=True):
    print(f"  {t}: {count:,} ({100*count/total_lines:.1f}%)")

print(f"\nmodel detection:")
for model, count in sorted(models_detected.items(), key=lambda x: x[1], reverse=True):
    print(f"  {model}: {count:,} ({100*count/total_lines:.1f}%)")

print(f"\ndate distribution:")
for date, count in sorted(dates.items()):
    print(f"  {date}: {count:,}")

