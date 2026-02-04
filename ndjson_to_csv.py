#!/usr/bin/env python3
import json
import csv

# paste your corpus filename here
CORPUS_FILE = "combined_corpus.ndjson"
OUTPUT_FILE = "corpus_readable.csv"

print(f"converting {CORPUS_FILE} to {OUTPUT_FILE}...")

with open(CORPUS_FILE, 'r', encoding='utf-8') as infile, \
     open(OUTPUT_FILE, 'w', encoding='utf-8', newline='') as outfile:
    
    writer = csv.writer(outfile)
    
    # write header
    writer.writerow([
        'Date',
        'Subreddit',
        'Author',
        'Comment',
        'Score',
        'Link'
    ])
    
    count = 0
    for line in infile:
        line = line.strip()
        if not line:
            continue
        
        try:
            data = json.loads(line)
            
            writer.writerow([
                data.get('created_date', ''),
                data.get('subreddit', ''),
                data.get('author', ''),
                data.get('body', ''),
                data.get('score', 0),
                data.get('full_link', '')
            ])
            
            count += 1
            if count % 10000 == 0:
                print(f"processed {count:,} records...")
                
        except json.JSONDecodeError:
            pass

print(f"\ndone ^_^ wrote {count:,} records to {OUTPUT_FILE}")