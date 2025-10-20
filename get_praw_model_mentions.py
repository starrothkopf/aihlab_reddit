import os
import praw
import re
import csv
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm  

load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("CLIENT_ID"),
    client_secret=os.getenv("CLIENT_SECRET"),
    password=os.getenv("PASSWORD"),
    user_agent=os.getenv("USER_AGENT"),
    username=os.getenv("USERNAME"),
)

MAX_ITEMS = None  
EXCLUDED_AUTHORS = {"AutoModerator", "ChatGPT-ModTeam"}
SEARCH_SUBREDDITS = ["ChatGPT", "technology", "artificial", "AI_Agents", "LLMDevs"] # "ArtificialIntelligence" banned??

# only include submissions/comments from 2025
START_2025 = datetime(2025, 1, 1).timestamp()
END_2025 = datetime(2025, 12, 31, 23, 59, 59).timestamp()

MODEL_PATTERNS = [
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
combined_model_regex = re.compile("|".join(MODEL_PATTERNS), re.IGNORECASE)

def process_submission(submission, writer):
    """Process a single submission: check title + selftext + comments"""
    collected = 0

    # skip if author is excluded
    if submission.author and submission.author.name in EXCLUDED_AUTHORS:
        return collected

    # skip submission if not in 2025
    if not (START_2025 <= submission.created_utc <= END_2025):
        return collected

    text_blocks = []
    if submission.title:
        text_blocks.append(submission.title)
    if hasattr(submission, "selftext") and submission.selftext:
        text_blocks.append(submission.selftext)

    for text in text_blocks:
        match = combined_model_regex.search(text)
        if match:
            writer.writerow([
                match.group(0).lower(),
                str(submission.subreddit),
                str(submission.author),
                datetime.utcfromtimestamp(submission.created_utc).strftime("%Y-%m-%d"),
                text.replace("\n"," ").replace("\r"," ")
            ])
            collected += 1

    # process comments
    submission.comments.replace_more(limit=0)
    for comment in submission.comments.list():
        if comment.author and comment.author.name in EXCLUDED_AUTHORS:
            continue
        if not (START_2025 <= comment.created_utc <= END_2025):
            continue

        match = combined_model_regex.search(comment.body)
        if match:
            writer.writerow([
                match.group(0).lower(),
                str(comment.subreddit),
                str(comment.author),
                datetime.utcfromtimestamp(comment.created_utc).strftime("%Y-%m-%d"),
                comment.body.replace("\n"," ").replace("\r"," ")
            ])
            collected += 1
        if MAX_ITEMS and collected >= MAX_ITEMS:
            break

    return collected

def main():
    report = {}
    for sub_name in SEARCH_SUBREDDITS:
        output_file = f"praw_{sub_name}_2025.csv"
        total_collected = 0

        with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["model_name", "subreddit", "user", "date", "raw_text"])

            subreddit = reddit.subreddit(sub_name)
            print(f"processing subreddit: {sub_name}")

            for submission in tqdm(subreddit.new(limit=MAX_ITEMS), desc=f"processing {sub_name}"):
                collected = process_submission(submission, writer)
                total_collected += collected
                if MAX_ITEMS and total_collected >= MAX_ITEMS:
                    break

        report[sub_name] = total_collected
        print(f"done ^_^ finished {sub_name}: {total_collected} mentions saved to {output_file}\n")

    print("summary report:")
    for sub_name, count in report.items():
        print(f"{sub_name}: {count} mentions")

if __name__ == "__main__":
    main()
