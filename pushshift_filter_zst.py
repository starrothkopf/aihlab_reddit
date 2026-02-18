import zstandard
import os
import json
from datetime import datetime
import logging
from collections import defaultdict
import re
from pathlib import Path

INPUT_FOLDER = "pushshift_25_07_zst_unfiltered/"  # folder containing all .zst files
OUTPUT_FOLDER = "pushshift_25_07_json/"  # output folder for processed files
START_YEAR = 2022

log = logging.getLogger("reddit_filter")
log.setLevel(logging.INFO)
handler = logging.StreamHandler()
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
handler.setFormatter(log_formatter)
log.addHandler(handler)

model_patterns = [
    r"gpt[-\s]?3\.?5",
    r"chat[\s-]?gpt[-\s]?3\.?5",
    r"gpt[-\s]?4o",
    r"chat[\s-]?gpt[-\s]?4o",
    r"gpt[-\s]?4",
    r"chat[\s-]?gpt[-\s]?4",
    r"gpt[-\s]?5",
    r"chat[\s-]?gpt[-\s]?5",
    r"\bchat[\s-]?gpt\b",
]

combined_model_regex = re.compile("|".join(model_patterns), re.IGNORECASE)

GPT4_RELEASE = datetime(2023, 3, 14)
GPT4O_RELEASE = datetime(2024, 5, 13)

def read_lines_zst(file_name):
    """get one line of json at a time from a zst file"""
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
    if not text:
        return None

    match = combined_model_regex.search(text)
    if not match:
        return None

    found = match.group(0).lower()

    if "3.5" in found:
        return "gpt-3.5"
    elif "4o" in found:
        return "gpt-4o"
    elif "4" in found:
        return "gpt-4"
    elif re.search(r"\b5\b|gpt[-\s]?5", found):
        return "gpt-5"
    else:
        return "chatgpt"


def extract_submission_metadata(obj):
    """get metadata from a submission"""
    return {
        "type": "submission",
        "id": obj.get("id"),
        "title": obj.get("title", ""),
        "selftext": obj.get("selftext", ""),
        "author": obj.get("author", "[deleted]"),
        "author_flair_text": obj.get("author_flair_text"),
        "created_utc": obj.get("created_utc"),
        "created_date": datetime.utcfromtimestamp(int(obj['created_utc'])).strftime("%Y-%m-%d %H:%M:%S"),
        "subreddit": obj.get("subreddit", ""),
        "score": obj.get("score", 0),
        "upvote_ratio": obj.get("upvote_ratio"),
        "num_comments": obj.get("num_comments", 0),
        "url": obj.get("url", ""),
        "permalink": obj.get("permalink", ""),
        "is_self": obj.get("is_self", False),
        "is_original_content": obj.get("is_original_content", False),
        "over_18": obj.get("over_18", False),
        "spoiler": obj.get("spoiler", False),
        "locked": obj.get("locked", False),
        "stickied": obj.get("stickied", False),
        "distinguished": obj.get("distinguished"),
        "edited": obj.get("edited", False),
        "link_flair_text": obj.get("link_flair_text"),
        "domain": obj.get("domain"),
        "gilded": obj.get("gilded", 0),
        "name": obj.get("name"),
        "full_link": f"https://reddit.com{obj.get('permalink', '')}" if obj.get('permalink') else None,
    }

def extract_comment_metadata(obj):
    """get metadata from a comment"""
    return {
        "type": "comment",
        "id": obj.get("id"),
        "link_id": obj.get("link_id"),  # submission ID (with prefix)
        "parent_id": obj.get("parent_id"),  # parent comment/submission ID
        "body": obj.get("body", ""),
        "author": obj.get("author", "[deleted]"),
        "author_flair_text": obj.get("author_flair_text"),
        "created_utc": obj.get("created_utc"),
        "created_date": datetime.utcfromtimestamp(int(obj['created_utc'])).strftime("%Y-%m-%d %H:%M:%S"),
        "subreddit": obj.get("subreddit", ""),
        "score": obj.get("score", 0),
        "edited": obj.get("edited", False),
        "distinguished": obj.get("distinguished"),
        "stickied": obj.get("stickied", False),
        "permalink": obj.get("permalink", ""),
        "is_submitter": obj.get("is_submitter", False),
        "controversiality": obj.get("controversiality", 0),
        "gilded": obj.get("gilded", 0),
        "name": obj.get("name"),
        "full_link": f"https://reddit.com{obj.get('permalink', '')}" if obj.get('permalink') else None,
    }


def process_file(input_file, output_file):
    """process a single zst and output matching entries to json"""
    total = 0
    matched = 0
    yearly_counts = defaultdict(lambda: defaultdict(int))
    chatgpt_period_counts = {"pre_gpt4": 0, "pre_gpt4o": 0, "post_gpt4o": 0}
    
    is_submission = "submission" in input_file.lower()
    
    log.info(f"processing {'submissions' if is_submission else 'comments'}: {input_file}")

    with open(output_file, 'w', encoding='utf-8') as jsonfile:
        for line in read_lines_zst(input_file):
            total += 1
            try:
                obj = json.loads(line)
                
                # skips
                author = obj.get("author", "").lower()
                if any(bad in author for bad in ["automoderator", "moderator", "modmail", "modteam"]):
                    continue

                created = datetime.utcfromtimestamp(int(obj['created_utc']))
                if created.year < START_YEAR:
                    continue

                if is_submission:
                    text_field = obj.get("title", "") + " " + obj.get("selftext", "")
                else:
                    text_field = obj.get("body", "")
                
                if not text_field or not text_field.strip():
                    continue

                model_found = contains_model(text_field)
                if model_found:
                    matched += 1
                    year = created.year
                    yearly_counts[year][model_found] += 1

                    # track temporal distribution
                    if model_found == "chatgpt":
                        if created < GPT4_RELEASE:
                            chatgpt_period_counts["pre_gpt4"] += 1
                        elif created < GPT4O_RELEASE:
                            chatgpt_period_counts["pre_gpt4o"] += 1
                        else:
                            chatgpt_period_counts["post_gpt4o"] += 1

                    if is_submission:
                        entry = extract_submission_metadata(obj)
                    else:
                        entry = extract_comment_metadata(obj)
                    
                    entry["model_detected"] = model_found
                    entry["detection_text"] = text_field[:500]  # store snippet for verification
                    
                    # write as ndjson (newline-delimited json)
                    jsonfile.write(json.dumps(entry) + '\n')

                if total % 100000 == 0:
                    log.info(f"processed {total:,} lines, matched {matched:,} lines")

            except (KeyError, json.JSONDecodeError, ValueError) as e:
                continue

    log.info(f"  complete: {input_file}")
    log.info(f"  total: {total:,}, matched: {matched:,}, output: {output_file}")
    log.info("  mentions per year and model:")
    for year in sorted(yearly_counts):
        for model, count in yearly_counts[year].items():
            log.info(f"    {year} - {model}: {count:,}")

    if any(chatgpt_period_counts.values()):
        log.info("  chatgpt temporal breakdown:")
        for period, count in chatgpt_period_counts.items():
            log.info(f"    {period}: {count:,}")
    
    return matched, yearly_counts, chatgpt_period_counts


def main():
    """process all zst files in the input folder"""
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
    input_path = Path(INPUT_FOLDER)
    zst_files = list(input_path.glob("*.zst"))

    log.info(f"found {len(zst_files)} ZST files")
    
    overall_stats = {
        "total_matched": 0,
        "yearly_counts": defaultdict(lambda: defaultdict(int)),
        "chatgpt_periods": {"pre_gpt4": 0, "pre_gpt4o": 0, "post_gpt4o": 0}
    }
    
    # process each file
    for zst_file in sorted(zst_files):
        base_name = zst_file.stem 
        output_file = Path(OUTPUT_FOLDER) / f"{base_name}_filtered.json"
        
        try:
            matched, yearly_counts, chatgpt_periods = process_file(
                str(zst_file), 
                str(output_file)
            )
            
            overall_stats["total_matched"] += matched
            for year, models in yearly_counts.items():
                for model, count in models.items():
                    overall_stats["yearly_counts"][year][model] += count
            for period, count in chatgpt_periods.items():
                overall_stats["chatgpt_periods"][period] += count
                
        except Exception as e:
            log.error(f"(!) error processing {zst_file}: {e}")
            continue
    
    log.info("\n" + "="*60)
    log.info("summary")
    log.info("="*60)
    log.info(f"total matched across all files: {overall_stats['total_matched']:,}")
    log.info("\ncombined mentions per year and model:")
    for year in sorted(overall_stats['yearly_counts']):
        for model, count in overall_stats['yearly_counts'][year].items():
            log.info(f"  {year} - {model}: {count:,}")
    
    log.info("\ncombined ChatGPT temporal breakdown:")
    for period, count in overall_stats['chatgpt_periods'].items():
        log.info(f"  {period}: {count:,}")
    
    log.info(f"\noutput files saved to: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()