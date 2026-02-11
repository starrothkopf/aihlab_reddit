import json
import re
from collections import defaultdict, Counter
from pathlib import Path
import logging
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

INPUT_FILE = "combined_corpus_with_deps_lg.ndjson"
OUTPUT_FILE = "all_adjectives_analysis.json"
NUM_PROCESSES = cpu_count() - 1  
CHUNK_SIZE = 10000  

log = logging.getLogger("adjective_analyzer")
log.setLevel(logging.INFO)
handler = logging.StreamHandler()
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
handler.setFormatter(log_formatter)
log.addHandler(handler)

MODEL_PATTERNS = {
    "gpt-3.5": re.compile(r"gpt[-\s]?3\.?5|chat[\s-]?gpt[-\s]?3\.?5", re.IGNORECASE),
    "gpt-4o": re.compile(r"gpt[-\s]?4o|chat[\s-]?gpt[-\s]?4o", re.IGNORECASE),
    "gpt-4": re.compile(r"gpt[-\s]?4(?!o)|chat[\s-]?gpt[-\s]?4(?!o)", re.IGNORECASE),
    "gpt-5": re.compile(r"gpt[-\s]?5(?!o)|chat[\s-]?gpt[-\s]?5(?!o)", re.IGNORECASE),
    "chatgpt": re.compile(r"\bchat[\s-]?gpt\b", re.IGNORECASE),
}

def find_models_in_text(text_lower):
    found = []
    for model_name, pattern in MODEL_PATTERNS.items():
        if pattern.search(text_lower):
            found.append(model_name)
    return found

def is_negated(token_data, sent_tokens):
    token_id = token_data.get('id')
    
    for other_token in sent_tokens:
        # check if this token is a negation word modifying target token
        if other_token.get('dep') == 'neg' and other_token.get('head') == token_id:
            return True
    return False


def extract_all_adjectives(spacy_parse):
    if not spacy_parse or not spacy_parse.get('sentences'):
        return []
    
    adjectives = []
    
    for sent in spacy_parse['sentences']:
        tokens = sent['tokens']
        
        for token_data in tokens:
            pos = token_data.get('pos', '')
            tag = token_data.get('tag', '')
            lemma = token_data.get('lemma', '').lower()
            text = token_data.get('text', '')
            dep = token_data.get('dep', '')
            
            # extract adjectives (ADJ pos tag)
            if pos == 'ADJ':
                negated = is_negated(token_data, tokens)
                adjectives.append({
                    'lemma': lemma,
                    'text': text,
                    'pos': pos,
                    'dep': dep,
                    'tag': tag,
                    'negated': negated
                })
            
            # include participles acting as adjectives
            elif pos == 'VERB' and tag in ['VBN', 'VBG'] and dep in ['amod', 'acl', 'relcl']:
                negated = is_negated(token_data, tokens)
                adjectives.append({
                    'lemma': lemma,
                    'text': text,
                    'pos': pos,
                    'dep': dep,
                    'tag': tag,
                    'negated': negated
                })
    
    return adjectives


def process_entry(line):
    try:
        entry = json.loads(line)
        
        if entry.get("type") == "submission":
            text = f"{entry.get('title', '')} {entry.get('selftext', '')}".strip()
        else:
            text = entry.get("body", "").strip()
        
        if not text:
            return None
        
        text_lower = text.lower()
        
        # find which models are mentioned
        models_found = find_models_in_text(text_lower)
        if not models_found:
            return None
        
        # get dependency parse
        spacy_parse = entry.get("spacy_parse")
        if not spacy_parse:
            return None
        
        # extract all adjectives
        adjectives = extract_all_adjectives(spacy_parse)
        
        if not adjectives:
            return None
        
        # return adjectives categorized by which models were mentioned
        return {
            'models': models_found,
            'adjectives': adjectives
        }
        
    except Exception:
        return None


def process_chunk(lines):
    chunk_results = {
        'model_adjectives': defaultdict(Counter),
        'model_adjectives_negated': defaultdict(Counter),
        'entries_processed': 0
    }
    
    for line in lines:
        result = process_entry(line)
        if result:
            chunk_results['entries_processed'] += 1
            adjectives = result['adjectives']
            models = result['models']
            
            for model in models:
                for adj in adjectives:
                    lemma = adj['lemma']
                    if adj.get('negated'):
                        chunk_results['model_adjectives_negated'][model][lemma] += 1
                    else:
                        chunk_results['model_adjectives'][model][lemma] += 1
    
    return chunk_results


def read_in_chunks(file_path, chunk_size):
    chunk = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunk.append(line)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


def main():
    log.info(f"processing file: {INPUT_FILE}")
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    log.info(f"total entries: {total_lines:,}")
    
    adjective_counts = defaultdict(Counter)
    negated_adjective_counts = defaultdict(Counter) 
    total_processed = 0
    
    with Pool(processes=NUM_PROCESSES) as pool:
        chunks = list(read_in_chunks(INPUT_FILE, CHUNK_SIZE))
        with tqdm(total=len(chunks), desc="processing chunks", unit="chunk") as pbar:
            for chunk_result in pool.imap_unordered(process_chunk, chunks):
                for model, counts in chunk_result['model_adjectives'].items():
                    adjective_counts[model].update(counts)
                
                for model, counts in chunk_result['model_adjectives_negated'].items():
                    negated_adjective_counts[model].update(counts)
                
                total_processed += chunk_result['entries_processed']
                pbar.update(1)
                pbar.set_postfix({'entries': total_processed})
    
    log.info(f"\nprocessed {total_processed:,} entries with model mentions")
    
    model_mentions = {model: sum(counts.values()) for model, counts in adjective_counts.items()}
    
    log.info(f"\nadjective mentions by model:")
    for model, count in sorted(model_mentions.items(), key=lambda x: x[1], reverse=True):
        log.info(f"  {model}: {count:,}")
    
    results = {
        "summary": {
            "total_entries_processed": total_processed,
            "model_adjective_mentions": model_mentions,
            "model_negated_adjective_mentions": {
                model: sum(counts.values()) 
                for model, counts in negated_adjective_counts.items()
            },
            "total_unique_adjectives": len(set().union(*[set(counts.keys()) for counts in adjective_counts.values()]))
        },
        "top_adjectives_by_model": {},
        "top_negated_adjectives_by_model": {}  
    }
    
    # positive and negated
    for model, counts in adjective_counts.items():
        if counts:
            results["top_adjectives_by_model"][model] = counts.most_common(500)
            log.info(f"\n{model.upper()} - Top 10 adjectives:")
            for adj, count in counts.most_common(10):
                log.info(f"  {adj}: {count:,}")
    
    for model, counts in negated_adjective_counts.items():
        if counts:
            results["top_negated_adjectives_by_model"][model] = counts.most_common(500)
            log.info(f"\n{model.upper()} - Top 10 NEGATED adjectives:")
            for adj, count in counts.most_common(10):
                log.info(f"  NOT {adj}: {count:,}")
    
    # save results
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    log.info(f"\nresults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()