import pandas as pd
from collections import Counter
from textblob import TextBlob

df = pd.read_csv("model_mentions.csv")

# drop rows without adjectives
df["Adjectives"] = df["Adjectives"].fillna("").astype(str)
# split adjectives into lists
df["Adjectives"] = df["Adjectives"].apply(lambda x: [adj.strip().lower() for adj in x.split(",") if adj.strip()])

# most common adjectives per model
def top_adjectives(model, n=10):
    adjectives = [adj for row in df[df["Model name"].str.upper() == model]["Adjectives"] for adj in row]
    counter = Counter(adjectives)
    return counter.most_common(n)

print("most common adjectives per model:")
for model in ["GPT-4", "GPT-5"]:
    print(f"\n{model}:")
    for adj, count in top_adjectives(model):
        print(f"  {adj}: {count}")

# sentiment analysis on adjectives
def sentiment_for_model(model):
    adjectives = [adj for row in df[df["Model name"].str.upper() == model]["Adjectives"] for adj in row]
    text = " ".join(adjectives)
    if not text:
        return 0
    return TextBlob(text).sentiment.polarity  # -1 (negative) → +1 (positive)

print("\nsentiment scores:")
for model in ["GPT-4", "GPT-5"]:
    score = sentiment_for_model(model)
    print(f"{model}: {score:.3f}")

# other sanity checks
print("\nother checks:")
print("total mentions by model:")
print(df["Model name"].value_counts())

print("\ntop 5 subreddits by mentions:")
print(df["Subreddit"].value_counts().head(5))

print("\ntop 5 users by mentions:")
print(df["User"].value_counts().head(5))

print("\ndate range of mentions:")
print(f"From {df['Date'].min()} to {df['Date'].max()}")
