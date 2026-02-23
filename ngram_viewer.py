import json
import re
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
import sys

# python ngram_viewer.py [word1] [word2] [word 3] ...

INPUT_FILE = "combined_corpus_cleaned_inferred_models.ndjson"
START_DATE = "2022-12"   

def get_text(entry):
    if entry.get("type") == "submission":
        return f"{entry.get('title','')} {entry.get('selftext','')}"
    return entry.get("body", "")

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def month_bucket(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    return dt.strftime("%Y-%m")  # monthly bucket

def generate_ngram_plot(words):
    words = [w.lower() for w in words]

    total_tokens_by_month = defaultdict(int)
    word_counts_by_month = {w: defaultdict(int) for w in words}

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)

            created = entry.get("created_date")
            if not created:
                continue

            month = month_bucket(created)
            tokens = tokenize(get_text(entry))

            total_tokens_by_month[month] += len(tokens)

            for w in words:
                word_counts_by_month[w][month] += tokens.count(w)

    months = sorted(
        m for m in total_tokens_by_month.keys()
        if m >= START_DATE
    )

    for w in words:
        frequencies = []
        for m in months:
            total = total_tokens_by_month[m]
            count = word_counts_by_month[w][m]
            freq = count / total if total > 0 else 0
            frequencies.append(freq)

        plt.plot(months, frequencies, label=w)

    plt.xticks(rotation=45)
    plt.ylabel("Relative Frequency (word count / total tokens)")
    plt.xlabel("Month")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ngram_viewer.py word1 word2 ...")
    else:
        generate_ngram_plot(sys.argv[1:])