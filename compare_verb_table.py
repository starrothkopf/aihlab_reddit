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

MODEL = "gpt-4"  # Can change to gpt-5, chatgpt, etc.
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
    Returns:
    - freq_dict_by[verb] = count (verbs BY model)
    - freq_dict_to[verb] = count (verbs TO model)
    - unique_by = {verb} (only in BY)
    - unique_to = {verb} (only in TO)
    - common = {verb} (in both)
    """

    freq_dict_by = {}
    freq_dict_to = {}
    
    # Get verbs done BY the model
    if model in results.get("top_verbs_by_model", {}):
        items_by = results["top_verbs_by_model"][model][:top_n]
        freq_dict_by = {verb: count for verb, count in items_by}
    
    # Get verbs done TO the model
    if model in results.get("top_verbs_to_model", {}):
        items_to = results["top_verbs_to_model"][model][:top_n]
        freq_dict_to = {verb: count for verb, count in items_to}
    
    set_by = set(freq_dict_by.keys())
    set_to = set(freq_dict_to.keys())
    
    common = set_by & set_to
    unique_by = set_by - set_to
    unique_to = set_to - set_by

    return freq_dict_by, freq_dict_to, unique_by, unique_to, common


def plot_verb_comparison(results, model, top_n):
    """
    Create a two-column comparison showing:
    - Left column: Top verbs done BY the model (model as agent)
    - Right column: Top verbs done TO the model (model as patient)
    - Highlight verbs that are unique to each side in top 20
    """
    freq_by, freq_to, unique_by, unique_to, common = prepare_verb_data(results, model, top_n)

    # Get top 20 verbs for each side
    top_by = sorted(freq_by.items(), key=lambda x: x[1], reverse=True)[:20]
    top_to = sorted(freq_to.items(), key=lambda x: x[1], reverse=True)[:20]
    
    # Identify which are unique in top 20
    top_by_verbs = set([v for v, _ in top_by])
    top_to_verbs = set([v for v, _ in top_to])
    
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 12))
    fig.patch.set_facecolor('white')
    
    # Left column: Verbs BY model
    ax_left.axis('off')
    ax_left.set_xlim(0, 1)
    ax_left.set_ylim(0, 1)
    
    # Title for left column
    ax_left.text(0.5, 0.95, f"Verbs BY {model.upper()}", 
                ha='center', fontsize=18, fontweight='bold', color="#1565C0")
    ax_left.text(0.5, 0.92, "(Model does action)", 
                ha='center', fontsize=12, style='italic', color="#555555")
    
    # Draw left column verbs
    y_pos = 0.87
    for i, (verb, count) in enumerate(top_by, 1):
        # Check if this verb is unique (not in top 20 of other side)
        is_unique = verb not in top_to_verbs
        
        # Different styling for unique vs common
        if is_unique:
            bgcolor = '#E3F2FD'  # Light blue highlight
            textcolor = '#0D47A1'  # Dark blue
            weight = 'bold'
        else:
            bgcolor = '#F5F5F5'  # Light gray
            textcolor = '#424242'  # Dark gray
            weight = 'normal'
        
        # Draw background box
        rect = plt.Rectangle((0.05, y_pos - 0.018), 0.9, 0.035, 
                            facecolor=bgcolor, edgecolor='#BDBDBD', linewidth=0.5)
        ax_left.add_patch(rect)
        
        # Draw rank, verb, and count
        ax_left.text(0.08, y_pos, f"{i}.", ha='left', fontsize=11, 
                    color=textcolor, weight=weight)
        ax_left.text(0.18, y_pos, verb, ha='left', fontsize=11, 
                    color=textcolor, weight=weight)
        ax_left.text(0.92, y_pos, f"{count:,}", ha='right', fontsize=11, 
                    color=textcolor, weight=weight)
        
        y_pos -= 0.042
    
    # Add legend for left column
    ax_left.text(0.5, 0.05, f"★ Highlighted = Unique to BY (not in top 20 TO)", 
                ha='center', fontsize=10, style='italic', color="#1565C0")
    ax_left.text(0.5, 0.02, f"Total unique BY: {len(unique_by)} | Common: {len(common)}", 
                ha='center', fontsize=9, color="#555555")
    
    # Right column: Verbs TO model
    ax_right.axis('off')
    ax_right.set_xlim(0, 1)
    ax_right.set_ylim(0, 1)
    
    # Title for right column
    ax_right.text(0.5, 0.95, f"Verbs TO {model.upper()}", 
                 ha='center', fontsize=18, fontweight='bold', color="#6A1B9A")
    ax_right.text(0.5, 0.92, "(Action done to model)", 
                 ha='center', fontsize=12, style='italic', color="#555555")
    
    # Draw right column verbs
    y_pos = 0.87
    for i, (verb, count) in enumerate(top_to, 1):
        # Check if this verb is unique (not in top 20 of other side)
        is_unique = verb not in top_by_verbs
        
        # Different styling for unique vs common
        if is_unique:
            bgcolor = '#F3E5F5'  # Light purple highlight
            textcolor = '#4A148C'  # Dark purple
            weight = 'bold'
        else:
            bgcolor = '#F5F5F5'  # Light gray
            textcolor = '#424242'  # Dark gray
            weight = 'normal'
        
        # Draw background box
        rect = plt.Rectangle((0.05, y_pos - 0.018), 0.9, 0.035, 
                            facecolor=bgcolor, edgecolor='#BDBDBD', linewidth=0.5)
        ax_right.add_patch(rect)
        
        # Draw rank, verb, and count
        ax_right.text(0.08, y_pos, f"{i}.", ha='left', fontsize=11, 
                     color=textcolor, weight=weight)
        ax_right.text(0.18, y_pos, verb, ha='left', fontsize=11, 
                     color=textcolor, weight=weight)
        ax_right.text(0.92, y_pos, f"{count:,}", ha='right', fontsize=11, 
                     color=textcolor, weight=weight)
        
        y_pos -= 0.042
    
    # Add legend for right column
    ax_right.text(0.5, 0.05, f"★ Highlighted = Unique to TO (not in top 20 BY)", 
                 ha='center', fontsize=10, style='italic', color="#6A1B9A")
    ax_right.text(0.5, 0.02, f"Total unique TO: {len(unique_to)} | Common: {len(common)}", 
                 ha='center', fontsize=9, color="#555555")
    
    # Overall title
    fig.suptitle(f"Verb Agency Analysis: {model.upper()}\nTop 20 Verbs (BY vs TO)", 
                fontsize=20, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = f"{OUTPUT_DIR}/verb_agency_comparison_{model.replace('-', '_')}.png"
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved: {out}")
    return out


def plot_all_models(results):
    """Generate verb comparison charts for all available models"""
    output_files = []
    
    # Get all models that have verb data
    models_with_data = set()
    if "top_verbs_by_model" in results:
        models_with_data.update(results["top_verbs_by_model"].keys())
    if "top_verbs_to_model" in results:
        models_with_data.update(results["top_verbs_to_model"].keys())
    
    print(f"Found {len(models_with_data)} models with verb data: {', '.join(sorted(models_with_data))}")
    
    for model in sorted(models_with_data):
        print(f"\nProcessing {model}...")
        output_file = plot_verb_comparison(results, model, TOP_N)
        output_files.append(output_file)
    
    return output_files


def print_summary_stats(results):
    """Print summary statistics about verbs"""
    print("\n" + "="*70)
    print("VERB ANALYSIS SUMMARY")
    print("="*70)
    
    if "summary" in results:
        summary = results["summary"]
        print(f"\nTotal entries with verbs: {summary.get('total_entries_with_verbs', 0):,}")
        print(f"Total unique verbs BY models: {summary.get('total_unique_verbs_by', 0):,}")
        print(f"Total unique verbs TO models: {summary.get('total_unique_verbs_to', 0):,}")
    
    print("\n" + "-"*70)
    print("VERBS BY MODEL (Model as agent - model does action)")
    print("-"*70)
    if "top_verbs_by_model" in results:
        for model, verbs in sorted(results["top_verbs_by_model"].items()):
            total = sum(count for _, count in verbs)
            unique = len(verbs)
            print(f"{model:12s}: {total:6,} mentions, {unique:4,} unique verbs")
            print(f"             Top 5: {', '.join([v for v, _ in verbs[:5]])}")
    
    print("\n" + "-"*70)
    print("VERBS TO MODEL (Model as patient - action done to model)")
    print("-"*70)
    if "top_verbs_to_model" in results:
        for model, verbs in sorted(results["top_verbs_to_model"].items()):
            total = sum(count for _, count in verbs)
            unique = len(verbs)
            print(f"{model:12s}: {total:6,} mentions, {unique:4,} unique verbs")
            print(f"             Top 5: {', '.join([v for v, _ in verbs[:5]])}")


def main():
    create_output_dir()
    results = load_results()
    
    print_summary_stats(results)
    
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    output_files = plot_all_models(results)
    
    print("\n" + "="*70)
    print(f"Complete! Generated {len(output_files)} visualization(s)")
    print("="*70)


if __name__ == "__main__":
    main()