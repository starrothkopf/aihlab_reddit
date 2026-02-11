import json
import re
import csv
from collections import defaultdict, Counter
from pathlib import Path
import logging

INPUT_FILE = "mini_corpus_with_scores_deps.ndjson" 
OUTPUT_FILE = "nouns_scores_inspection.csv"

log = logging.getLogger("noun_analyzer")
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


def build_dependency_structures(full_tree):
    token_info = {}
    for token in full_tree:
        idx = token.get('idx') or token.get('token_idx')
        token_info[idx] = token
    
    # children by parent (for traversing tree)
    children_by_parent = defaultdict(list)
    
    for token in full_tree:
        idx = token.get('idx') or token.get('token_idx')
        head_idx = token.get('head_idx')
        
        # build parent-child relationships
        if head_idx is not None and head_idx != idx:  # avoid self-loops
            children_by_parent[head_idx].append(idx)
    
    return {
        'token_info': token_info,
        'children_by_parent': children_by_parent
    }


def find_model_token_indices_enhanced(full_tree, text_lower, model_names):
    model_tokens = defaultdict(set)
    text_to_tokens = defaultdict(list)
    for i, token in enumerate(full_tree):
        idx = token.get('idx') or token.get('token_idx', i)
        token_text_lower = token.get('text', '').lower()
        text_to_tokens[token_text_lower].append(idx)
    
    # for each model, find all tokens that could be part of the mention
    for model_name in model_names:
        # keywords that indicate this model
        if 'gpt' in model_name:
            keywords = ['gpt', 'chatgpt']
            if '3.5' in model_name or '3' in model_name:
                keywords.extend(['3.5', '3', 'three'])
            elif '4o' in model_name:
                keywords.extend(['4o', '4', 'four'])
            elif '4' in model_name:
                keywords.extend(['4', 'four'])
            elif '5' in model_name:
                keywords.extend(['5', 'five'])
        else:
            keywords = [model_name.lower()]
        
        # find all tokens containing these keywords
        for keyword in keywords:
            for token_text, indices in text_to_tokens.items():
                if keyword in token_text or token_text in keyword:
                    model_tokens[model_name].update(indices)
        
        # also check for compound noun phrases
        for i, token in enumerate(full_tree):
            idx = token.get('idx') or token.get('token_idx', i)
            head_idx = token.get('head_idx')
            dep = token.get('dep', '')
            
            if idx in model_tokens[model_name]:
                # include compound and flat dependencies
                for j, other in enumerate(full_tree):
                    other_idx = other.get('idx') or other.get('token_idx', j)
                    other_head = other.get('head_idx')
                    other_dep = other.get('dep', '')
                    
                    if other_head == idx and other_dep in ['compound', 'flat', 'nummod']:
                        model_tokens[model_name].add(other_idx)
                    elif head_idx == other_idx and dep in ['compound', 'flat', 'nummod']:
                        model_tokens[model_name].add(other_idx)
    
    # convert sets to lists
    return {model: list(indices) for model, indices in model_tokens.items()}


def extract_nouns_comprehensive(structures, model_token_indices):
    nouns = []
    token_info = structures['token_info']
    children_by_parent = structures['children_by_parent']
    
    for token_idx in model_token_indices:
        if token_idx not in token_info:
            continue
        
        token = token_info[token_idx]
        head_idx = token.get('head_idx')
        dep = token.get('dep')
        pos = token.get('pos')
        
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
        
        # Look for compound modifiers of the model
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
    
    # filter out self-references (nouns that are actually model names)
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


def process_entry(entry, entry_id):
    try:
        if entry.get("type") == "submission":
            text = f"{entry.get('title', '')} {entry.get('selftext', '')}".strip()
            entry_type = "submission"
        else:
            text = entry.get("body", "").strip()
            entry_type = "comment"
        
        if not text:
            return None
        
        text_lower = text.lower()
        
        models_found = find_models_in_text(text_lower)
        if not models_found:
            return None
        
        # get dependency parse data
        dep_parse = entry.get("dependency_parse")
        if not dep_parse or not dep_parse.get('full_tree'):
            return None
        
        # get the full tree directly
        full_tree = dep_parse['full_tree']
        
        # ensure each token has 'idx' field
        for i, token in enumerate(full_tree):
            if 'idx' not in token:
                token['idx'] = i
        
        # build dependency structures
        structures = build_dependency_structures(full_tree)
        
        # find model tokens
        model_tokens = find_model_token_indices_enhanced(full_tree, text_lower, models_found)
        
        created_date = entry.get('created_date', '')
        is_anthro1 = entry.get('IsAnthro1', '')
        is_anthro2 = entry.get('IsAnthro2', '')
        
        results = []
        
        for model_name in models_found:
            token_indices = model_tokens.get(model_name, [])
            if not token_indices:
                continue
            
            # extract nouns
            nouns = extract_nouns_comprehensive(structures, token_indices)
            
            for noun in nouns:
                results.append({
                    'entry_id': entry_id,
                    'entry_type': entry_type,
                    'created_date': created_date,
                    'text': text[:1000],  # truncate long texts
                    'model': model_name,
                    'noun_lemma': noun['lemma'],
                    'noun_text': noun['text'],
                    'dependency_relation': noun['dep'],
                    'pos': noun['pos'],
                    'pattern': noun.get('pattern', 'unknown'),
                    'IsAnthro1': is_anthro1,
                    'IsAnthro2': is_anthro2
                })
        
        return results
        
    except Exception as e:
        log.error(f"error processing entry {entry_id}: {e}")
        return None


def main():
    log.info(f"processing file: {INPUT_FILE}")
    
    csv_rows = []
    entries_processed = 0
    entries_with_nouns = 0
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for entry_id, line in enumerate(f, 1):
            try:
                entry = json.loads(line)
                entries_processed += 1
                
                results = process_entry(entry, entry_id)
                if results:
                    entries_with_nouns += 1
                    csv_rows.extend(results)
                    
                if entries_processed % 100 == 0:
                    log.info(f"processed {entries_processed} entries, found {entries_with_nouns} with nouns")
                    
            except json.JSONDecodeError:
                log.error(f"JSON decode error on line {entry_id}")
                continue
    
    log.info(f"\ntotal processed: {entries_processed}")
    log.info(f"entries with nouns: {entries_with_nouns}")
    log.info(f"total noun extractions: {len(csv_rows)}")
    
    # write to CSV
    if csv_rows:
        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['entry_id', 'entry_type', 'created_date', 'model', 'noun_lemma', 
                         'noun_text', 'dependency_relation', 'pos', 'pattern', 'text', 'IsAnthro1', 'IsAnthro2']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in csv_rows:
                writer.writerow(row)
        
        log.info(f"\nresults saved to: {OUTPUT_FILE}")
        
        # some stats
        patterns = Counter(row['pattern'] for row in csv_rows)
        log.info("\nextraction patterns used:")
        for pattern, count in patterns.most_common():
            log.info(f"  {pattern}: {count}")
        
        nouns = Counter(row['noun_lemma'] for row in csv_rows)
        log.info("\ntop 20 nouns found:")
        for noun, count in nouns.most_common(20):
            log.info(f"  {noun}: {count}")
    else:
        log.info("no nouns found!")


if __name__ == "__main__":
    main()