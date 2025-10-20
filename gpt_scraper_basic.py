import os
import praw
import re
import csv
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from dotenv import load_dotenv
import spacy
from tqdm import tqdm  

"""
strategy shifted, now separating scraping from parsing/analysis for easier iteration
"""

# downloads
nltk.download("punkt")
nltk.download("stopwords")

load_dotenv()
reddit = praw.Reddit(
    client_id=os.getenv("CLIENT_ID"),
    client_secret=os.getenv("CLIENT_SECRET"),
    password=os.getenv("PASSWORD"),
    user_agent=os.getenv("USER_AGENT"),
    username=os.getenv("USERNAME"),
)

MAX_ITEMS = 100
OUTPUT_FILE = "model_mentions.csv"
NEGATORS = {"not", "never", "hardly", "no", "barely"}

# spaCy pipeline (disable NER for speed)
nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
pattern = re.compile(r"\bgpt[- ]?4(o)?\b|\bgpt[- ]?5(o)?\b", re.IGNORECASE)
stop_words = set(stopwords.words("english"))

def extract_adjectives(doc, model_name):
    """extract adjectives/phrases that describe GPT models."""
    results = []
    skip_words = {"gpt", "chatgpt", "openai", "dalle", "x1f916", "gpt5", "gpt4", "prompt"}

    for token in doc:
        if token.text.lower() in [model_name.lower(), "gpt-4","4o","gpt-5","5o"]:
            for child in token.children:
                if child.dep_ in {"acomp", "attr"} and child.pos_ in {"ADJ","NOUN","PROPN"}:
                    phrase = " ".join([w.text for w in child.subtree])
                    if phrase.lower() not in skip_words:
                        results.append(phrase.lower())
    return results

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Model name", "Adjectives", "Subreddit", "User", "Date", "Raw text"])

    subreddit = reddit.subreddit("ChatGPT")
    collected = 0

    # wrap with tqdm to show live progress toward MAX_ITEMS
    for submission in subreddit.top(time_filter="month", limit=None):
        if submission.author and submission.author.name in ["AutoModerator", "ChatGPT-ModTeam"]:
            continue

        submission.comments.replace_more(limit=0)
        comments = submission.comments.list()

        for comment in tqdm(comments, desc="Processing comments", total=len(comments)):
            if comment.author and comment.author.name in ["AutoModerator", "ChatGPT-ModTeam"]:
                continue

            sentences = nltk.sent_tokenize(comment.body)
            # use nlp.pipe for all candidate sentences (faster than looping)
            matches = [(s, pattern.search(s)) for s in sentences]
            batch = [s for s, m in matches if m]

            for doc, (sentence, match) in zip(nlp.pipe(batch), [(s, m) for s,m in matches if m]):
                model_name = match.group(0).upper()
                adjectives = extract_adjectives(doc, model_name)

                if adjectives:
                    print(f"✅ Found: {model_name} -> {adjectives}")
                    date_str = datetime.utcfromtimestamp(comment.created_utc).strftime("%m/%d/%Y")
                    writer.writerow([
                        model_name,
                        ", ".join(adjectives),
                        str(comment.subreddit),
                        str(comment.author),
                        date_str,
                        sentence.strip()
                    ])
                    collected += 1
                    if collected >= MAX_ITEMS:
                        break
            if collected >= MAX_ITEMS:
                break
        if collected >= MAX_ITEMS:
            break

print(f"done ^_^ data saved to {OUTPUT_FILE}")
