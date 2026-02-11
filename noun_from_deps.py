import json
import re
from collections import defaultdict
from pathlib import Path
import logging
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

INPUT_FILE = "combined_corpus_with_deps_lg.ndjson"
OUTPUT_FILE = "model_nouns_corpus.ndjson"
NUM_PROCESSES = cpu_count() - 1
CHUNK_SIZE = 10000

log = logging.getLogger("noun_analyzer")
log.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
handler.setFormatter(formatter)
log.addHandler(handler)

MODEL_PATTERNS = {
    "gpt-3.5": re.compile(r"gpt[-\s]?3\.?5|chat[\s-]?gpt[-\s]?3\.?5", re.IGNORECASE),
    "gpt-4o": re.compile(r"gpt[-\s]?4o|chat[\s-]?gpt[-\s]?4o", re.IGNORECASE),
    "gpt-4": re.compile(r"gpt[-\s]?4(?!o)|chat[\s-]?gpt[-\s]?4(?!o)", re.IGNORECASE),
    "gpt-5": re.compile(r"gpt[-\s]?5(?!o)|chat[\s-]?gpt[-\s]?5(?!o)", re.IGNORECASE),
    "chatgpt": re.compile(r"\bchat[\s-]?gpt\b", re.IGNORECASE),
}

def find_models_in_text(text_lower):
    return [
        model for model, pattern in MODEL_PATTERNS.items()
        if pattern.search(text_lower)
    ]

def build_dependency_structures(full_tree):
    token_info = {}
    children_by_parent = defaultdict(list)

    for token in full_tree:
        idx = token.get('idx') or token.get('token_idx')
        token_info[idx] = token

    for token in full_tree:
        idx = token.get('idx') or token.get('token_idx')
        head_idx = token.get('head_idx')
        if head_idx is not None and head_idx != idx:
            children_by_parent[head_idx].append(idx)

    return {
        "token_info": token_info,
        "children_by_parent": children_by_parent
    }

def find_model_token_indices(full_tree, text_lower, model_names):
    model_tokens = defaultdict(set)

    for token in full_tree:
        idx = token.get('idx') or token.get('token_idx')
        token_text = token.get("text", "").lower()

        for model in model_names:
            if MODEL_PATTERNS[model].search(token_text):
                model_tokens[model].add(idx)

    return {k: list(v) for k, v in model_tokens.items()}

def extract_associated_nouns(structures, model_token_indices):
    nouns = []
    token_info = structures["token_info"]
    children_by_parent = structures["children_by_parent"]

    for token_idx in model_token_indices:
        if token_idx not in token_info:
            continue
        
        token = token_info[token_idx]
        head_idx = token.get("head_idx")
        dep = token.get("dep")
        pos = token.get("pos")

        # PATTERN 1: Copular predicate
        # "ChatGPT is a tool", "GPT-4 seems like a chatbot"
        if dep in ['nsubj', 'nsubjpass']:
            if head_idx and head_idx in token_info:
                verb = token_info[head_idx]
                verb_lemma = verb.get('lemma', '').lower()
                
                # Check for copular verbs
                if verb_lemma in ['be', 'seem', 'feel']:
                    # Look for attribute nouns
                    for child_idx in children_by_parent.get(head_idx, []):
                        child = token_info.get(child_idx)
                        if child and child.get('dep') in ['attr', 'acomp'] and child.get('pos') in ['NOUN', 'PROPN']:
                            nouns.append({
                                'lemma': child.get('lemma', '').lower(),
                                'text': child.get('text', ''),
                                'dep': child.get('dep'),
                                'pos': child.get('pos'),
                                'pattern': 'copular_predicate'
                            })
        
        
        # PATTERN 2: Prepositional comparisons
        # "like ChatGPT", "as a tool"
        if dep in ['pobj', 'obj']:
            if head_idx and head_idx in token_info:
                prep = token_info[head_idx]
                prep_lemma = prep.get('lemma', '').lower()
                
                if prep_lemma in ['like', 'as']:
                    if pos in ['NOUN', 'PROPN']:
                        nouns.append({
                            'lemma': token.get('lemma', '').lower(),
                            'text': token.get('text', ''),
                            'dep': dep,
                            'pos': pos,
                            'pattern': 'prepositional_comparison'
                        })
        
        # PATTERN 3: Compound nouns
        # "chatbot GPT-4", "AI assistant ChatGPT"
        # Model is part of a compound
        if dep in ['compound', 'flat']:
            if head_idx and head_idx in token_info:
                head_token = token_info[head_idx]
                if head_token.get('pos') in ['NOUN', 'PROPN']:
                    nouns.append({
                        'lemma': head_token.get('lemma', '').lower(),
                        'text': head_token.get('text', ''),
                        'dep': head_token.get('dep'),
                        'pos': head_token.get('pos'),
                        'pattern': 'compound_head'
                    })
        
        # look for compound modifiers of the model
        for child_idx in children_by_parent.get(token_idx, []):
            child = token_info.get(child_idx)
            if child and child.get('dep') in ['compound', 'flat'] and child.get('pos') in ['NOUN', 'PROPN']:
                nouns.append({
                    'lemma': child.get('lemma', '').lower(),
                    'text': child.get('text', ''),
                    'dep': child.get('dep'),
                    'pos': child.get('pos'),
                    'pattern': 'compound_modifier'
                })
        
        
        for child_idx in children_by_parent.get(token_idx, []):
            child = token_info.get(child_idx)
            if child and child.get('dep') == 'conj' and child.get('pos') in ['NOUN', 'PROPN']:
                nouns.append({
                    'lemma': child.get('lemma', '').lower(),
                    'text': child.get('text', ''),
                    'dep': child.get('dep'),
                    'pos': child.get('pos'),
                    'pattern': 'conjoined_noun'
                })

    filtered_nouns = []
    for noun in nouns:
        noun_lower = noun['lemma'].lower()
        noun_text_lower = noun['text'].lower()
        
        is_model_name = any(pattern.search(noun_lower) for pattern in MODEL_PATTERNS.values())
        
        contains_gpt_or_chat = 'gpt' in noun_lower or 'chat' in noun_lower or \
                               'gpt' in noun_text_lower or 'chat' in noun_text_lower
        
        if not is_model_name and not contains_gpt_or_chat:
            filtered_nouns.append(noun)
    
    return filtered_nouns

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
        models_found = find_models_in_text(text_lower)
        if not models_found:
            return None

        spacy_parse = entry.get("spacy_parse")
        if not spacy_parse or not spacy_parse.get("sentences"):
            return None

        full_tree = []
        for sent in spacy_parse["sentences"]:
            for token_data in sent["tokens"]:
                token_data["idx"] = token_data["token_idx"]
                full_tree.append(token_data)

        structures = build_dependency_structures(full_tree)
        model_tokens = find_model_token_indices(full_tree, text_lower, models_found)

        results = []

        for model in models_found:
            indices = model_tokens.get(model, [])
            if not indices:
                continue

            nouns = extract_associated_nouns(structures, indices)
            if nouns:
                # Extract just the lemmas like the adjective code does
                output_entry = {
                    "id": entry.get("id"),
                    "model_detected": model,
                    "created_date": entry.get("created_date"),
                    "text": text,
                    "nouns": [noun['lemma'] for noun in nouns]  # just the lemmas
                }
                results.append(output_entry)

        return results if results else None

    except Exception:
        return None

def process_chunk(lines):
    results = []
    count = 0

    for line in lines:
        r = process_entry(line)
        if r:
            count += len(r)
            results.extend(r)

    return results, count

def read_in_chunks(file_path, chunk_size):
    chunk = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            chunk.append(line)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

def main():
    log.info(f"processing file: {INPUT_FILE}")
    log.info(f"using {NUM_PROCESSES} processes")
    log.info("counting entries...")
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    log.info(f"total entries: {total_lines:,}")

    total_processed = 0

    log.info("processing entries in parallel...")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        with Pool(processes=NUM_PROCESSES) as pool:
            chunks = list(read_in_chunks(INPUT_FILE, CHUNK_SIZE))

            with tqdm(total=len(chunks), desc="processing chunks", unit="chunk") as pbar:
                for chunk_results, entries_in_chunk in pool.imap_unordered(process_chunk, chunks):
                    for result in chunk_results:
                        outfile.write(json.dumps(result, ensure_ascii=False) + "\n")

                    total_processed += entries_in_chunk
                    pbar.update(1)
                    pbar.set_postfix({'entries_with_nouns': total_processed})

    log.info(f"\nprocessed {total_processed:,} entries with nouns")
    log.info(f"\n ^_^ results saved to: {OUTPUT_FILE}")

    log.info("\ncounting noun frequencies...")
    from collections import Counter
    global_noun_counts = Counter()
    
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            for noun in entry.get("nouns", []):
                global_noun_counts[noun] += 1
    
    log.info("\ntop 30 nouns overall:")
    for noun, count in global_noun_counts.most_common(30):
        log.info(f"  {noun}: {count:,}")


if __name__ == "__main__":
    main()