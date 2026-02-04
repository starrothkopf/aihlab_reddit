# Reddit ChatGPT Discourse Corpus Pipeline
This repository contains the data collection pipeline used to build a corpus of 870,364 Reddit posts and comments mentioning ChatGPT models from 2022–2025. The dataset itself (several GB) is hosted on Box—this repo documents it was built and how it can be replicated.

Dataset Access: https://wustl.box.com/s/m5gcaktpkp425967efffni89yccaybsd

**Corpus Summary**
Total records: 870,364
Total file size (chars): 1,257,523,965 (~1.2GB)
Total body text (chars): 263,399,485

Types:
  comment: 649,203 (74.6%)
  submission: 221,161 (25.4%)

Model detection:
  chatgpt: 718,643 (82.6%)
  gpt-4: 79,992 (9.2%)
  gpt-5: 41,292 (4.7%)
  gpt-4o: 18,325 (2.1%)
  gpt-3.5: 12,112 (1.4%)
Source Subreddits: r/ChatGPT, r/technology, r/ArtificialIntelligence, r/artificial

Each entry is a newline-delimited JSON object with 38 fields:
Core Identification:

type: "submission" or "comment"
id: Reddit unique identifier
created_utc, created_date: Unix timestamp and human-readable datetime
author: Username (or "[deleted]")
subreddit: Source community

Content:

title, selftext: Post title and body (submissions)
body: Comment text
text_for_count: Combined text field for analysis
word_count: Token count

Engagement Metrics:

score: Net upvotes
upvote_ratio: Percentage upvoted (submissions only)
num_comments: Discussion volume (submissions only)
controversiality: Reddit's controversy flag (comments only)
gilded: Premium awards received

Thread Structure:

link_id: Parent submission ID (comments only)
parent_id: Parent comment/submission ID (comments only)
is_submitter: Whether commenter is the OP

Model Detection (Added by Pipeline):

model_detected: Which model was mentioned (gpt-3.5, gpt-4, gpt-4o, gpt-5, or chatgpt)
detection_text: First 500 characters showing context of mention

Other Metadata:

author_flair_text, link_flair_text: User/post flair
url, permalink, full_link: Post URLs
distinguished, stickied, locked: Moderator actions
edited, over_18, spoiler: Content flags
year_month, year: Temporal groupings for analysis

### Main Scripts

**pushshift_combine_folder_multiprocess.py**
Extracts subreddit-specific data from Pushshift's massive monthly dumps. This is necessary for 2025 data, which isn't pre-separated by subreddit. Decompresses ZST files, filters by subreddit field, outputs separate ZST files per subreddit, tracks progress and saves state (resumable if interrupted)

Usage:
bashpython3 pushshift_combine_folder_multiprocess.py [input_folder] \
  --value ChatGPT,technology,ArtificialIntelligence,artificial \
  --output [output_folder] \
  --processes 6

Important: Delete the pushshift_working folder before processing a new batch, or you'll get mismatched args errors.

**pushshift_filter_zst.py**
The meat! Scans subreddit ZSTs for ChatGPT model mentions using regex, tags ambiguous mentions based on release dates, outputs newline-delimited JSON for easy downstream processing. Regex patterns match variations: gpt-4, gpt 4, GPT4, chat-gpt-4, etc. Skips automoderator posts. 

Modify INPUT_FOLDER and OUTPUT_FOLDER constants in the script, then run for logs showing yearly mention counts and temporal breakdowns.

**Utility Scripts**

**plot_mentions_over_time.py**
Simple matplotlib graph comparing model mention frequency over time. Useful for checking data integrity and spotting temporal patterns.

**praw_scrape_filter.py**
Not recommended. Initial attempt using Reddit's Python API Wrapper. Hit rate limits quickly and produced data biased toward popular posts. Keeping for documentation purposes. The Pushshift approach is far more comprehensive.

### Want to collect data for Claude, Gemini, or other models? 
Step 1: Get 2022-2024 Data
Torrent from the pre-separated archive (easiest): [https://academictorrents.com/details/1614740ac8c94505e4ecb9d88be8bed7b6afddd4](https://academictorrents.com/details/1614740ac8c94505e4ecb9d88be8bed7b6afddd4)
Search for your target subreddits (e.g., "r/claude"), download both submission (RS_) and comment (RC_) ZST files. This goes fast, subreddit-specific files are much smaller than full dumps

Step 2: Get 2025 Data
Torrent from monthly full-Reddit dumps: [https://academictorrents.com/details/30dee5f0406da7a353aff6a8caa2d54fd01f2ca1](https://academictorrents.com/details/1614740ac8c94505e4ecb9d88be8bed7b6afddd4)
Download monthly ZSTs (~50GB each), don't process all months at once or you'll max out your CPU. Do them sequentially.
Run the multiprocess script per month:
bashpython3 pushshift_combine_folder_multiprocess.py [month_folder] \
  --value claude,ClaudeAI,anthropic \
  --output [destination] \
  --processes 6

**Remember:** Delete pushshift_working/ before processing each new month.

Step 3: Filter for Specific Models
Adapt pushshift_filter_zst.py for your target:
- Modify the model_patterns regex list
- Update contains_model() detection logic
- Adjust release date constants if you're tracking version rollouts
- Run on your extracted subreddit ZSTs


## Acknowledgments

- **u/Watchful1** for maintaining Pushshift torrents post-2023
- **u/RaiderBDev** for 2025 Reddit crawls
- **Pushshift** for archiving Reddit data (RIP easy API access)
- The pushshift_combine_folder_multiprocess.py script is mostly from RaiderBDev's processing tools
