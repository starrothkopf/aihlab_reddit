import json
import re
import hashlib
import logging
from pathlib import Path

INPUT_FILE = "combined_corpus.ndjson"
OUTPUT_FILE = "combined_corpus_cleaned.ndjson"

# replace vs remove platform artifacts
REPLACE_USERS = True
REPLACE_SUBREDDITS = True

log = logging.getLogger("reddit_cleaner")
log.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
log.addHandler(handler)

URL_RE = re.compile(r"http\S+|www\S+", re.I)
USER_RE = re.compile(r"\bu\/[A-Za-z0-9_-]+", re.I)
SUB_RE = re.compile(r"\br\/[A-Za-z0-9_-]+", re.I)

# common reddit bot signatures
BOT_SIG_RE = re.compile(
    r"i am a bot.*?performed automatically.*",
    re.I | re.S
)

def clean_text(text): # minimal
    if not text:
        return None

    text = text.strip()

    if not text:
        return None

    # replace urls
    text = URL_RE.sub("_URL_", text)

    # replace user mentions
    if REPLACE_USERS:
        text = USER_RE.sub("_USER_", text)

    # replace subreddit mentions
    if REPLACE_SUBREDDITS:
        text = SUB_RE.sub("_SUBREDDIT_", text)

    # remove bot signatures
    text = BOT_SIG_RE.sub("", text)

    if not text:
        return None

    return text

TEXT_FIELDS = [
    "body",
    "title",
    "selftext",
    "detection_text",
]

def clean_entry(obj):
    has_valid_text = False
    combined_text = []

    for field in TEXT_FIELDS:
        if field in obj and obj[field]:
            cleaned = clean_text(obj[field])

            if cleaned is None:
                obj[field] = ""
            else:
                obj[field] = cleaned
                has_valid_text = True
                combined_text.append(cleaned)

    if not has_valid_text:
        return None

    return obj, " ".join(combined_text)

def process_file(input_file, output_file):

    total = 0
    kept = 0
    removed = 0

    log.info(f"reading: {input_file}")

    with open(input_file, "r", encoding="utf-8") as infile, \
         open(output_file, "w", encoding="utf-8") as outfile:

        for line in infile:
            total += 1

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            result = clean_entry(obj)

            if result is None:
                removed += 1
                continue

            outfile.write(json.dumps(obj) + "\n")
            kept += 1

            if total % 100000 == 0:
                log.info(
                    f"processed {total:,}, kept {kept:,}, removed {removed:,}"
                )

    log.info(f"^_^ done")
    log.info(f"output: {output_file}")

if __name__ == "__main__":
    process_file(INPUT_FILE, OUTPUT_FILE)
