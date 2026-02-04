import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import spacy
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


RESULTS_FILE = "model_adjectives_analysis_lg.json"
OUTPUT_DIR = "visualizations"

MODELS = ["gpt-4", "gpt-5"]
TOP_N = 100
WORDS_PER_REGION = 20

MIN_FONT = 12
MAX_FONT = 20

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 12)
plt.rcParams["font.size"] = 10


nlp = spacy.load("en_core_web_lg")
nltk.download("vader_lexicon") 
sia = SentimentIntensityAnalyzer()


def load_results():
    with open(RESULTS_FILE, "r") as f:
        return json.load(f)

def create_output_dir():
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

def scale_font_sizes(counts):
    """map counts to font sizes"""
    counts = np.array(counts)
    if counts.max() == counts.min():
        return [MIN_FONT] * len(counts)

    return MIN_FONT + (counts - counts.min()) / (counts.max() - counts.min()) * (MAX_FONT - MIN_FONT)

def sentiment_score(words):
    if not words:
        return 0.0

    scores = [sia.polarity_scores(w)["compound"] for w in words]
    return round(np.mean(scores), 3)


def prepare_adjective_data(results, models, top_n):
    """
    returns:
    - freq_dict[model][adj] = count
    - unique[model] = {adj}
    - common = {adj}
    """

    freq_dict = {}
    sets = {}

    for model in models:
        items = results["top_adjectives_by_model"][model][:top_n]
        freq_dict[model] = {adj: count for adj, count in items}
        sets[model] = set(freq_dict[model].keys())

    common = sets[models[0]] & sets[models[1]]
    unique = {
        models[0]: sets[models[0]] - sets[models[1]],
        models[1]: sets[models[1]] - sets[models[0]],
    }

    return freq_dict, unique, common


def draw_word_block(ax, words, freqs, center, title=None):
    counts = [freqs[w] for w in words]
    font_sizes = scale_font_sizes(counts)

    y_positions = np.linspace(center[1] + 0.25, center[1] - 0.25, len(words))

    for word, count, fs, y in zip(words, counts, font_sizes, y_positions):
        ax.text(
            center[0],
            y,
            f"{word} ({count})",
            ha="center",
            va="center",
            fontsize=fs,
        )

    if title:
        ax.text(center[0], center[1] + 0.32, title, ha="center", fontsize=11, fontweight="bold")

def plot_venn(results, models, top_n):
    freq_dict, unique, common = prepare_adjective_data(results, models, top_n)

    fig, ax = plt.subplots()

    ax.add_patch(plt.Circle((0.35, 0.5), 0.32, alpha=0.25, color="#2E86AB"))
    ax.add_patch(plt.Circle((0.65, 0.5), 0.32, alpha=0.25, color="#A23B72"))

    left_words = sorted(
        unique[models[0]],
        key=lambda w: freq_dict[models[0]][w],
        reverse=True
    )[:WORDS_PER_REGION]

    right_words = sorted(
        unique[models[1]],
        key=lambda w: freq_dict[models[1]][w],
        reverse=True
    )[:WORDS_PER_REGION]

    common_words = sorted(
        common,
        key=lambda w: min(freq_dict[models[0]][w], freq_dict[models[1]][w]),
        reverse=True
    )[:WORDS_PER_REGION]

    draw_word_block(
        ax,
        left_words,
        freq_dict[models[0]],
        center=(0.22, 0.5),
        title=f"{models[0].upper()} unique"
    )

    draw_word_block(
        ax,
        right_words,
        freq_dict[models[1]],
        center=(0.78, 0.5),
        title=f"{models[1].upper()} unique"
    )

    common_freqs = {
        w: int((freq_dict[models[0]][w] + freq_dict[models[1]][w]) / 2)
        for w in common_words
    }

    draw_word_block(
        ax,
        common_words,
        common_freqs,
        center=(0.5, 0.5),
        title="Common"
    )

    s_left = sentiment_score(left_words)
    s_right = sentiment_score(right_words)

    ax.text(0.35, 0.15, f"{models[0].upper()} sentiment: {s_left}", ha="center", fontsize=14)
    ax.text(0.65, 0.15, f"{models[1].upper()} sentiment: {s_right}", ha="center", fontsize=14)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_aspect("equal")

    plt.title(
        f"Adjective Overlap with Frequency & Sentiment\nTop {top_n} adjectives",
        fontsize=15,
        pad=20
    )

    plt.tight_layout()
    out = f"{OUTPUT_DIR}/02_venn_weighted_frequency_sentiment_lg.png"
    plt.savefig(out, dpi=300)
    plt.close()

    print(out)

def main():
    create_output_dir()
    results = load_results()
    plot_venn(results, MODELS, TOP_N)

if __name__ == "__main__":
    main()
