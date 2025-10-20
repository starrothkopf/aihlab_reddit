import zstandard
import os
import json
import csv
from datetime import datetime
import logging
from collections import defaultdict
import re

input_file = "reddit/subreddits24/artificial_submissions.zst" # update as needed
output_file = "output_data/pushshift_artificial_posts.csv" # update as needed
output_format = "csv"

log = logging.getLogger("reddit_filter")
log.setLevel(logging.INFO)
handler = logging.StreamHandler()
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
handler.setFormatter(log_formatter)
log.addHandler(handler)

search_fields = ["body", "selftext", "title"]

# regex for flexible model and chatgpt references
model_patterns = [
    r"gpt[-\s]?3\.?5",     
    r"chat[\s-]?gpt[-\s]?3\.?5",
    r"gpt[-\s]?4o",        
    r"chat[\s-]?gpt[-\s]?4o",
    r"gpt[-\s]?4",          
    r"chat[\s-]?gpt[-\s]?4",
    r"gpt[-\s]?5o",
    r"chat[\s-]?gpt[-\s]?5o",
    r"gpt[-\s]?5",
    r"chat[\s-]?gpt[-\s]?5",
    r"\bchat[\s-]?gpt\b",  
]
combined_model_regex = re.compile("|".join(model_patterns), re.IGNORECASE)

GPT4_RELEASE = datetime(2023, 3, 14)
GPT4O_RELEASE = datetime(2024, 5, 13)
START_YEAR = 2022

def read_lines_zst(file_name):
    """Generator that yields one line of JSON at a time from a .zst file"""
    with open(file_name, 'rb') as fh:
        dctx = zstandard.ZstdDecompressor(max_window_size=2**31)
        with dctx.stream_reader(fh) as reader:
            buffer = b""
            while True:
                chunk = reader.read(2**27)  # 128 MB chunks
                if not chunk:
                    break
                buffer += chunk
                lines = buffer.split(b'\n')
                for line in lines[:-1]:
                    yield line.decode('utf-8', errors='ignore')
                buffer = lines[-1]
            if buffer:
                yield buffer.decode('utf-8', errors='ignore')


def contains_model(text): 
    """Normalize & detect any model mention or ChatGPT variant."""
    match = combined_model_regex.search(text)
    if not match:
        return None
    found = match.group(0).lower()
    if "5" in found:
        return "gpt-5"
    elif "4o" in found:
        return "gpt-4o"
    elif "4" in found:
        return "gpt-4"
    elif "3.5" in found:
        return "gpt-3.5"
    else:
        return "chatgpt"

def process_file(input_file, output_file):
    total = 0
    matched = 0
    yearly_counts = defaultdict(lambda: defaultdict(int))  # {year: {model: count}}
    chatgpt_period_counts = {"pre_gpt4": 0, "pre_gpt4o": 0, "post_gpt4o": 0}

    with open(output_file, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["subreddit", "user", "date", "model", "text"])  # header

        for line in read_lines_zst(input_file):
            total += 1
            try:
                obj = json.loads(line)
                author = obj.get("author", "").lower()
                if any(bad in author for bad in ["automoderator", "bot", "moderator", "modmail"]):
                    continue

                created = datetime.utcfromtimestamp(int(obj['created_utc']))

                if created.year < START_YEAR: # skip pre-2022
                    continue

                text_field = None
                for field in search_fields:
                    if field in obj and obj[field]:
                        text_field = obj[field]
                        break
                if not text_field:
                    continue

                model_found = contains_model(text_field)
                if model_found:
                    matched += 1
                    year = created.year
                    yearly_counts[year][model_found] += 1

                    # track time-based ChatGPT-only distribution
                    if model_found == "chatgpt":
                        if created < GPT4_RELEASE:
                            chatgpt_period_counts["pre_gpt4"] += 1
                        elif created < GPT4O_RELEASE:
                            chatgpt_period_counts["pre_gpt4o"] += 1
                        else:
                            chatgpt_period_counts["post_gpt4o"] += 1

                    writer.writerow([
                        obj.get("subreddit", ""),
                        f"u/{obj.get('author','')}",
                        created.strftime("%Y-%m-%d"),
                        model_found,
                        text_field.replace("\n", " ").replace("\r", " ")
                    ])

                if total % 100000 == 0:
                    log.info(f"processed {total:,} lines, matched {matched:,} lines")

            except (KeyError, json.JSONDecodeError):
                continue

    log.info(f"done ^_^ processed {total:,} lines, matched {matched:,}. output: {output_file}")
    log.info("mentions per year and model:")
    for year in sorted(yearly_counts):
        for model, count in yearly_counts[year].items():
            log.info(f"  {year} - {model}: {count:,}")

    if any(chatgpt_period_counts.values()):
        log.info("\nChatGPT-only time breakdown:")
        for period, count in chatgpt_period_counts.items():
            log.info(f"  {period}: {count:,}")


if __name__ == "__main__":
    process_file(input_file, output_file)