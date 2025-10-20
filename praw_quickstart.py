import os
from dotenv import load_dotenv
import praw

load_dotenv()  # environment variables from .env for safety :)

reddit = praw.Reddit(
    client_id=os.getenv("CLIENT_ID"),
    client_secret=os.getenv("CLIENT_SECRET"),
    password=os.getenv("PASSWORD"),
    user_agent=os.getenv("USER_AGENT"),
    username=os.getenv("USERNAME"),
)

print(reddit.read_only)
print(reddit.user.me())

for submission in reddit.subreddit("test").hot(limit=10):
    print(submission.title)

# "In the above example, we are limiting the results to 10. 
# Without the limit parameter PRAW should yield as many results as it can with a single request. 
# For most endpoints this results in 100 items per request. 
# If you want to retrieve as many as possible pass in limit=None."

subreddit = reddit.subreddit("redditdev")

print(subreddit.display_name)
# Output: redditdev
print(subreddit.title)
# Output: reddit development
print(subreddit.description)

for submission in subreddit.hot(limit=10):
    print(submission.title)
    # Output: the submission's title
    print(submission.score)
    # Output: the submission's score
    print(submission.id)
    # Output: the submission's ID
    print(submission.url)
    # Output: the URL the submission points to or the submission's URL if it's a self post
    redditor1 = submission.author
    print(redditor1.name)
    # Output: name of the redditor
    # assume you have a praw.Reddit instance bound to variable `reddit`
    redditor2 = reddit.redditor("bboe")
    print(redditor2.link_karma)
    # Output: u/bboe's karma

# assume you have a praw.Reddit instance bound to variable `reddit`
top_level_comments = list(submission.comments)
all_comments = submission.comments.list()
submission = reddit.submission("39zje0")
print(submission.title)  # to make it non-lazy
print.pprint(vars(submission))

"""
reddit will soon only be available over HTTPS
{'_additional_fetch_params': {},
 '_comments': <praw.models.comment_forest.CommentForest object at 0x10fe14fd0>,
 '_comments_by_id': {'t1_cs7vwlm': Comment(id='cs7vwlm'),
                     't1_cs7xcx2': Comment(id='cs7xcx2'),
                     't1_cs7ykx6': Comment(id='cs7ykx6'),
                     't1_cs81mem': Comment(id='cs81mem'),
                     't1_cs81xp8': Comment(id='cs81xp8'),
                     't1_cs82epc': Comment(id='cs82epc'),
                     't1_csow6zt': Comment(id='csow6zt'),
                     't1_csoycql': Comment(id='csoycql'),
                     't1_csp2uvk': Comment(id='csp2uvk'),
                     't1_csrxl8t': Comment(id='csrxl8t'),
                     't1_csrxya2': Comment(id='csrxya2'),
                     't1_cswg4ku': Comment(id='cswg4ku'),
                     't1_cu8kmrn': Comment(id='cu8kmrn'),
                     't1_cuawtww': Comment(id='cuawtww'),
                     't1_cucnyw8': Comment(id='cucnyw8'),
                     't1_cw9j1ef': Comment(id='cw9j1ef'),
                     't1_cwe9ki7': Comment(id='cwe9ki7')},
 '_fetched': True,
 '_reddit': <praw.reddit.Reddit object at 0x10e0c4bd0>,
 'all_awardings': [],
 'allow_live_comments': True,
 'approved_at_utc': None,
 'approved_by': None,
 'archived': True,
 'author': Redditor(name='rram'),
 'author_flair_background_color': None,
 'author_flair_css_class': None,
 'author_flair_richtext': [],
 'author_flair_template_id': None,
 'author_flair_text': None,
 'author_flair_text_color': None,
 'author_flair_type': 'text',
 'author_fullname': 't2_5wfps',
 'author_is_blocked': False,
 'author_patreon_flair': False,
 'author_premium': False,
 'awarders': [],
 'banned_at_utc': None,
 'banned_by': None,
 'can_gild': False,
 'can_mod_post': False,
 'category': None,
 'clicked': False,
 'comment_limit': 2048,
 'comment_sort': 'confidence',
 'content_categories': None,
 'contest_mode': False,
 'created': 1434418540.0,
 'created_utc': 1434418540.0,
 'discussion_type': None,
 'distinguished': 'admin',
 'domain': 'self.redditdev',
 'downs': 0,
 'edited': 1440173665.0,
 'gilded': 0,
 'gildings': {},
 'hidden': False,
 'hide_score': False,
 'id': '39zje0',
 'is_created_from_ads_ui': False,
 'is_crosspostable': True,
 'is_meta': False,
 'is_original_content': False,
 'is_reddit_media_domain': False,
 'is_robot_indexable': True,
 'is_self': True,
 'is_video': False,
 'likes': None,
 'link_flair_background_color': None,
 'link_flair_css_class': '',
 'link_flair_richtext': [],
 'link_flair_text': 'Reddit API',
 'link_flair_text_color': None,
 'link_flair_type': 'text',
 'locked': False,
 'media': None,
 'media_embed': {},
 'media_only': False,
 'mod_note': None,
 'mod_reason_by': None,
 'mod_reason_title': None,
 'mod_reports': [],
 'name': 't3_39zje0',
 'no_follow': False,
 'num_comments': 117,
 'num_crossposts': 0,
 'num_duplicates': 0,
 'num_reports': None,
 'over_18': False,
 'permalink': '/r/redditdev/comments/39zje0/reddit_will_soon_only_be_available_over_https/',
 'pinned': False,
 'post_hint': 'self',
 'preview': {'enabled': False,
             'images': [{'id': 'mKvBKwqPFmnxiYtLQRehhGDWhnrZdJVqzSL_7jJsHb4',
                         'resolutions': [{'height': 150,
                                          'url': 'https://external-preview.redd.it/L5CgcQzm_oDfAOyXjrsyqxB1cQW9Htc8VyqhoD0wrPU.jpg?width=108&crop=smart&auto=webp&s=4c9874a596b313db7111a5b5e194708dafcf3442',
                                          'width': 108}],
                         'source': {'height': 200,
                                    'url': 'https://external-preview.redd.it/L5CgcQzm_oDfAOyXjrsyqxB1cQW9Htc8VyqhoD0wrPU.jpg?auto=webp&s=600472675b48c5bc261ffab506d0ff52817f3ed6',
                                    'width': 144},
                         'variants': {}}]},
 'pwls': 6,
 'quarantine': False,
 'removal_reason': None,
 'removed_by': None,
 'removed_by_category': None,
 'report_reasons': None,
 'saved': False,
 'score': 275,
 'secure_media': None,
 'secure_media_embed': {},
 'selftext': 'Nearly 1 year ago we [gave you the ability to view reddit '
             'completely over '
             'SSL](http://www.redditblog.com/2014/09/hell-its-about-time-reddit-now-supports.html). '
             "Now we're ready to enforce that everyone use a secure connection "
             'with reddit.\n'
             '\n'
             '**Please ensure that all of your scripts can perform all of '
             'their functions over HTTPS by June 29.** At this time we will '
             'begin redirecting all site traffic to be over HTTPS and HTTP '
             'will no longer be available.\n'
             '\n'
             'If this will be a problem for you, please let us know '
             'immediately.\n'
             '\n'
             '**EDIT** 2015-08-21: IT IS DONE. You also have HSTS too.',
 'selftext_html': '<!-- SC_OFF --><div class="md"><p>Nearly 1 year ago we <a '
                  'href="http://www.redditblog.com/2014/09/hell-its-about-time-reddit-now-supports.html">gave '
                  'you the ability to view reddit completely over SSL</a>. Now '
                  'we&#39;re ready to enforce that everyone use a secure '
                  'connection with reddit.</p>\n'
                  '\n'
                  '<p><strong>Please ensure that all of your scripts can '
                  'perform all of their functions over HTTPS by June '
                  '29.</strong> At this time we will begin redirecting all '
                  'site traffic to be over HTTPS and HTTP will no longer be '
                  'available.</p>\n'
                  '\n'
                  '<p>If this will be a problem for you, please let us know '
                  'immediately.</p>\n'
                  '\n'
                  '<p><strong>EDIT</strong> 2015-08-21: IT IS DONE. You also '
                  'have HSTS too.</p>\n'
                  '</div><!-- SC_ON -->',
 'send_replies': True,
 'spoiler': False,
 'stickied': False,
 'subreddit': Subreddit(display_name='redditdev'),
 'subreddit_id': 't5_2qizd',
 'subreddit_name_prefixed': 'r/redditdev',
 'subreddit_subscribers': 79348,
 'subreddit_type': 'public',
 'suggested_sort': None,
 'thumbnail': 'self',
 'thumbnail_height': None,
 'thumbnail_width': None,
 'title': 'reddit will soon only be available over HTTPS',
 'top_awarded_type': None,
 'total_awards_received': 0,
 'treatment_tags': [],
 'ups': 275,
 'upvote_ratio': 0.97,
 'url': 'https://www.reddit.com/r/redditdev/comments/39zje0/reddit_will_soon_only_be_available_over_https/',
 'user_reports': [],
 'view_count': None,
 'visited': False,
 'wls': 6}
"""