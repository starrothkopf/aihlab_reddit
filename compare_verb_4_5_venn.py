import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import spacy
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


RESULTS_FILE = "model_verbs_analysis_lg.json"
OUTPUT_DIR = "visualizations"

TOP_N = 100
WORDS_PER_REGION = 20

MIN_FONT = 12
MAX_FONT = 20

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 12)
plt.rcParams["font.size"] = 10


nlp = spacy.load("en_core_web_lg")
nltk.download("vader_lexicon", quiet=True) 
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


def prepare_verb_data(results, model, top_n):
    """
    returns:
    - freq_dict_by[verb] = count (verbs BY model)
    - freq_dict_to[verb] = count (verbs TO model)
    - unique_by = {verb} (only in BY)
    - unique_to = {verb} (only in TO)
    - common = {verb} (in both)
    """

    freq_dict_by = {}
    freq_dict_to = {}
    
    # get verbs done BY the model
    if model in results.get("top_verbs_by_model", {}):
        items_by = results["top_verbs_by_model"][model][:top_n]
        freq_dict_by = {verb: count for verb, count in items_by}
    
    # get verbs done TO the model
    if model in results.get("top_verbs_to_model", {}):
        items_to = results["top_verbs_to_model"][model][:top_n]
        freq_dict_to = {verb: count for verb, count in items_to}
    
    set_by = set(freq_dict_by.keys())
    set_to = set(freq_dict_to.keys())
    
    common = set_by & set_to
    unique_by = set_by - set_to
    unique_to = set_to - set_by

    return freq_dict_by, freq_dict_to, unique_by, unique_to, common


def draw_word_block(ax, words, freqs, center, title=None, color="#000000"):
    """draw a block of words with varying font sizes based on frequency"""
    if not words:
        return
    
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
            color=color
        )

    if title:
        ax.text(
            center[0], 
            center[1] + 0.32, 
            title, 
            ha="center", 
            fontsize=11, 
            fontweight="bold",
            color=color
        )


def plot_verb_venn(results, model, top_n):
    """
    create a Venn diagram showing:
    - left circle: verbs done BY the model (model as agent)
    - right circle: verbs done TO the model (model as patient)
    - middle: verbs appearing in both categories
    """
    freq_by, freq_to, unique_by, unique_to, common = prepare_verb_data(results, model, top_n)

    fig, ax = plt.subplots(figsize=(16, 12))

    # left circle (BY model) - blue/teal
    ax.add_patch(plt.Circle((0.35, 0.5), 0.32, alpha=0.25, color="#2E86AB"))
    # right circle (TO model) - purple/magenta
    ax.add_patch(plt.Circle((0.65, 0.5), 0.32, alpha=0.25, color="#A23B72"))

    # get top words for each region
    left_words = sorted(
        unique_by,
        key=lambda w: freq_by[w],
        reverse=True
    )[:WORDS_PER_REGION]

    right_words = sorted(
        unique_to,
        key=lambda w: freq_to[w],
        reverse=True
    )[:WORDS_PER_REGION]

    # for common words, use average frequency for sorting
    common_words = sorted(
        common,
        key=lambda w: (freq_by.get(w, 0) + freq_to.get(w, 0)) / 2,
        reverse=True
    )[:WORDS_PER_REGION]

    # draw left region (verbs BY model)
    draw_word_block(
        ax,
        left_words,
        freq_by,
        center=(0.22, 0.5),
        title=f"BY {model.upper()}\n(Model does action)",
        color="#1a5276"
    )

    # draw right region (verbs TO model)
    draw_word_block(
        ax,
        right_words,
        freq_to,
        center=(0.78, 0.5),
        title=f"TO {model.upper()}\n(Action done to model)",
        color="#702963"
    )

    # draw middle region (verbs in both)
    # use average frequency for display
    common_freqs = {
        w: int((freq_by.get(w, 0) + freq_to.get(w, 0)) / 2)
        for w in common_words
    }

    draw_word_block(
        ax,
        common_words,
        common_freqs,
        center=(0.5, 0.5),
        title="BOTH\n(Bidirectional)",
        color="#4a235a"
    )

    # calculate sentiment scores
    s_by = sentiment_score(left_words)
    s_to = sentiment_score(right_words)
    s_common = sentiment_score(common_words)

    # add sentiment annotations at bottom
    ax.text(
        0.35, 0.12, 
        f"BY {model.upper()} sentiment: {s_by}", 
        ha="center", 
        fontsize=13,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#d6eaf8", alpha=0.7)
    )
    
    ax.text(
        0.65, 0.12, 
        f"TO {model.upper()} sentiment: {s_to}", 
        ha="center", 
        fontsize=13,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5b7b1", alpha=0.7)
    )
    
    ax.text(
        0.5, 0.05, 
        f"BOTH sentiment: {s_common}", 
        ha="center", 
        fontsize=13,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#e8daef", alpha=0.7)
    )

    # add counts at top
    ax.text(
        0.5, 0.92,
        f"Total unique BY: {len(unique_by)} | Total unique TO: {len(unique_to)} | Common: {len(common)}",
        ha="center",
        fontsize=12,
        style="italic",
        color="#555555"
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_aspect("equal")

    plt.title(
        f"Verb Agency Analysis: {model.upper()}\nTop {top_n} verbs (BY model vs TO model)",
        fontsize=16,
        pad=20,
        fontweight="bold"
    )

    plt.tight_layout()
    out = f"{OUTPUT_DIR}/verb_agency_venn_{model.replace('-', '_')}.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {out}")
    return out


def plot_all_models(results):
    """generate verb Venn diagrams for all available models"""
    output_files = []
    
    # get all models that have verb data
    models_with_data = set()
    if "top_verbs_by_model" in results:
        models_with_data.update(results["top_verbs_by_model"].keys())
    if "top_verbs_to_model" in results:
        models_with_data.update(results["top_verbs_to_model"].keys())
    
    print(f"found {len(models_with_data)} models with verb data: {', '.join(sorted(models_with_data))}")
    
    for model in sorted(models_with_data):
        print(f"\nprocessing {model}...")
        output_file = plot_verb_venn(results, model, TOP_N)
        output_files.append(output_file)
    
    return output_files


def main():
    create_output_dir()
    results = load_results()
    
    output_files = plot_all_models(results)
    
    print(f"complete ^_^ generated {len(output_files)} visualizations")


if __name__ == "__main__":
    main()