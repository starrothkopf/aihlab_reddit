import os
import praw
import re
import json
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm
import time

load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("CLIENT_ID"),
    client_secret=os.getenv("CLIENT_SECRET"),
    password=os.getenv("PASSWORD"),
    user_agent=os.getenv("USER_AGENT"),
    username=os.getenv("USERNAME"),
)

# dates: June 1 - December 31, 2025
START_DATE = datetime(2025, 6, 1).timestamp()
END_DATE = datetime(2025, 12, 31, 23, 59, 59).timestamp()

EXCLUDED_AUTHORS = {"AutoModerator", "ChatGPT-ModTeam"}
SEARCH_SUBREDDITS = ["ChatGPT", "technology", "artificial", "AI_Agents", "LLMDevs"]

# STAGE 1: broad Reddit search queries, cast a wide net to find candidate posts, relevance bias
SEARCH_QUERIES = [
    "chatgpt OR \"chat gpt\"",                   # general ChatGPT
    "\"3.5\" OR gpt-3.5 OR gpt3.5",              # GPT-3.5
    "(gpt-4 OR gpt4) NOT 4o",                    # GPT-4 (exclude 4o)
    "4o OR gpt-4o",                              # GPT-4o
    "(gpt-5 OR gpt5) NOT 5o",                    # GPT-5 (exclude 5o)
    "5o OR gpt-5o",                              # GPT-5o
]

# STAGE 2: precise regex patterns for confirmation after search finds candidates
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

def extract_submission_data(submission):
    """extract metadata from a submission (post)"""
    return {
        "type": "submission",
        "id": submission.id,
        "title": submission.title,
        "selftext": submission.selftext if hasattr(submission, "selftext") else "",
        "author": str(submission.author) if submission.author else "[deleted]",
        "author_flair_text": submission.author_flair_text,
        "created_utc": submission.created_utc,
        "created_date": datetime.utcfromtimestamp(submission.created_utc).strftime("%Y-%m-%d %H:%M:%S"),
        "subreddit": str(submission.subreddit),
        "score": submission.score,
        "upvote_ratio": submission.upvote_ratio,
        "num_comments": submission.num_comments,
        "url": submission.url,
        "permalink": f"https://reddit.com{submission.permalink}",
        "is_self": submission.is_self,
        "is_original_content": submission.is_original_content,
        "over_18": submission.over_18,
        "spoiler": submission.spoiler,
        "locked": submission.locked,
        "stickied": submission.stickied,
        "distinguished": submission.distinguished,
        "edited": submission.edited,
        "link_flair_text": submission.link_flair_text,
        "clicked": submission.clicked,
        "saved": submission.saved,
        "name": submission.name,
    }


def extract_comment_data(comment, submission_id):
    """extract metadata from a comment"""
    return {
        "type": "comment",
        "id": comment.id,
        "submission_id": submission_id,
        "parent_id": comment.parent_id,
        "body": comment.body,
        "author": str(comment.author) if comment.author else "[deleted]",
        "author_flair_text": comment.author_flair_text if hasattr(comment, "author_flair_text") else None,
        "created_utc": comment.created_utc,
        "created_date": datetime.utcfromtimestamp(comment.created_utc).strftime("%Y-%m-%d %H:%M:%S"),
        "subreddit": str(comment.subreddit),
        "score": comment.score,
        "edited": comment.edited,
        "distinguished": comment.distinguished,
        "stickied": comment.stickied,
        "permalink": f"https://reddit.com{comment.permalink}",
        "is_submitter": comment.is_submitter,
        "name": comment.name,
    }


def find_model_mentions(text):
    """find all model mentions in text using regex, return list of matched strings"""
    if not text:
        return []
    matches = combined_model_regex.findall(text)
    return [m.lower() for m in matches] if matches else []

def process_submission(submission, stats):
    """process a submission and its comments, returns list of items with model mentions, updates stats dict with processing info"""
    items = []
    
    # skips
    if submission.author and submission.author.name in EXCLUDED_AUTHORS:
        stats['excluded_authors'] += 1
        return items
    if not (START_DATE <= submission.created_utc <= END_DATE):
        stats['outside_date_range'] += 1
        return items
    
    # STAGE 2 FILTER: check submission with regex
    submission_text = f"{submission.title} {submission.selftext if hasattr(submission, 'selftext') else ''}"
    submission_mentions = find_model_mentions(submission_text)
    
    if submission_mentions:
        sub_data = extract_submission_data(submission)
        sub_data["model_mentions"] = submission_mentions
        items.append(sub_data)
        stats['submissions_with_mentions'] += 1
    else:
        stats['submissions_no_mentions'] += 1
    
    # process comments
    try:
        stats['total_declared_comments'] += submission.num_comments
        
        # fetches all comments (slow)
        submission.comments.replace_more(limit=None)
        
        comment_count_processed = 0
        comment_count_with_mentions = 0
        
        for comment in submission.comments.list():
            comment_count_processed += 1
            
            # skips
            if comment.author and comment.author.name in EXCLUDED_AUTHORS:
                continue
            if not (START_DATE <= comment.created_utc <= END_DATE):
                continue
            
            # STAGE 2 FILTER: check comment with regex
            comment_mentions = find_model_mentions(comment.body)
            if comment_mentions:
                comment_data = extract_comment_data(comment, submission.id)
                comment_data["model_mentions"] = comment_mentions
                items.append(comment_data)
                comment_count_with_mentions += 1
        
        stats['total_comments_processed'] += comment_count_processed
        stats['comments_with_mentions'] += comment_count_with_mentions
        
    except Exception as e:
        stats['comment_errors'] += 1
        print(f"(!) error processing comments for {submission.id}: {e}")
    
    return items


def search_subreddit_with_queries(subreddit, start_ts, end_ts, queries, limit_per_query=1000):
    """STAGE 1: use Reddit search with multiple queries to find candidate posts, returns unique submissions that match search"""
    seen_ids = set()
    all_submissions = []
    
    print(f"\nSTAGE 1: running {len(queries)} search queries...")
    
    for i, query in enumerate(queries, 1):
        print(f"    [{i}/{len(queries)}] query: '{query}'")
        try:
            # use Reddit search with time_filter='year'
            results = subreddit.search(query, time_filter='year', limit=limit_per_query, sort='new')
            
            count = 0
            for submission in results:
                # only add if in date range and not already seen
                if start_ts <= submission.created_utc <= end_ts and submission.id not in seen_ids:
                    seen_ids.add(submission.id)
                    all_submissions.append(submission)
                    count += 1
            
            print(f"              → found {count} unique submissions in date range")
            
            # rate limit protection
            time.sleep(2)
            
        except Exception as e:
            print(f"              (!) error: {e}")
            continue
    
    print(f"\n  STAGE 1 complete: {len(all_submissions)} unique candidate submissions found")
    return all_submissions


def main():
    overall_report = {}
    overall_stats = {
        'total_items': 0,
        'total_submissions': 0,
        'total_comments': 0,
    }
    
    for sub_name in SEARCH_SUBREDDITS:
        output_file = f"praw_{sub_name}_june_dec_2025_v3.json"
        all_items = []
        
        # track detailed stats for this subreddit
        stats = {
            'search_candidates': 0,
            'excluded_authors': 0,
            'outside_date_range': 0,
            'submissions_with_mentions': 0,
            'submissions_no_mentions': 0,
            'comments_with_mentions': 0,
            'total_declared_comments': 0,
            'total_comments_processed': 0,
            'comment_errors': 0,
        }
        
        print(f"\n{'='*70}")
        print(f"processing subreddit: r/{sub_name}")
        print(f"{'='*70}")
        
        try:
            subreddit = reddit.subreddit(sub_name)
            
            # STAGE 1: search for candidate posts
            submissions = search_subreddit_with_queries(
                subreddit, START_DATE, END_DATE, SEARCH_QUERIES, limit_per_query=1000
            )
            stats['search_candidates'] = len(submissions)
            
            # STAGE 2: process each submission with regex filtering
            print(f"\n  STAGE 2: processing {len(submissions)} candidates with regex filter...")
            print(f"              (this may take a bit!)\n")
            
            for i, submission in enumerate(tqdm(submissions, desc=f"  r/{sub_name}")):
                items_found = process_submission(submission, stats)
                all_items.extend(items_found)
                
            # save
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(all_items, f, ensure_ascii=False, indent=2)
            
            # metrics
            submissions_count = sum(1 for item in all_items if item["type"] == "submission")
            comments_count = sum(1 for item in all_items if item["type"] == "comment")
            
            capture_rate = (
                (stats['comments_with_mentions'] / stats['total_declared_comments'] * 100)
                if stats['total_declared_comments'] > 0 else 0
            )
            
            # stage 2 filter efficiency
            stage2_efficiency = (
                (stats['submissions_with_mentions'] / stats['search_candidates'] * 100)
                if stats['search_candidates'] > 0 else 0
            )
            
            overall_report[sub_name] = {
                "file": output_file,
                "total_items": len(all_items),
                "submissions": submissions_count,
                "comments": comments_count,
                "stats": stats,
                "capture_rate": round(capture_rate, 1),
                "stage2_efficiency": round(stage2_efficiency, 1),
            }
            
            overall_stats['total_items'] += len(all_items)
            overall_stats['total_submissions'] += submissions_count
            overall_stats['total_comments'] += comments_count
            
            # summary per subreddit
            print(f"\n  {'='*66}")
            print(f"  completed r/{sub_name}")
            print(f"  {'='*66}")
            print(f"  saved to: {output_file}")
            print(f"\n  RESULTS:")
            print(f"     total items with mentions: {len(all_items)}")
            print(f"     - submissions: {submissions_count}")
            print(f"     - comments: {comments_count}")
            print(f"\n  STAGE 1 (Search):")
            print(f"     candidates found: {stats['search_candidates']}")
            print(f"\n  STAGE 2 (Regex Filter):")
            print(f"     submissions with mentions: {stats['submissions_with_mentions']} ({stage2_efficiency}% of candidates)")
            print(f"     submissions without mentions: {stats['submissions_no_mentions']}")
            print(f"\n  COMMENTS:")
            print(f"     declared in submissions: {stats['total_declared_comments']}")
            print(f"     actually processed: {stats['total_comments_processed']}")
            print(f"     with model mentions: {stats['comments_with_mentions']}")
            print(f"     capture rate: {capture_rate:.1f}%")
            if stats['comment_errors'] > 0:
                print(f"     errors: {stats['comment_errors']}")
            
        except Exception as e:
            print(f"\n  (!) error processing r/{sub_name}: {e}")
            overall_report[sub_name] = {"error": str(e)}
    
    # final summary across all subreddits
    print(f"\n{'='*70}")
    print("== summary ==")
    print(f"{'='*70}")
    
    for sub_name, data in overall_report.items():
        if "error" in data:
            print(f"\nr/{sub_name}: error - {data['error']}")
        else:
            print(f"\nr/{sub_name}:")
            print(f"  total: {data['total_items']} items ({data['submissions']} posts, {data['comments']} comments)")
            print(f"  search efficiency: {data['stage2_efficiency']}% of candidates had mentions")
            print(f"  comment capture: {data['capture_rate']}%")
    
    print(f"\n{'='*70}")
    print(f"total: {overall_stats['total_items']} items")
    print(f"  submissions: {overall_stats['total_submissions']}")
    print(f"  comments: {overall_stats['total_comments']}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()