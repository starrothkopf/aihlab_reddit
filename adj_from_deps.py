import json
import re
from collections import defaultdict, Counter
from pathlib import Path
import logging
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import functools

INPUT_FILE = "combined_corpus_with_deps_lg.ndjson"
OUTPUT_FILE = "model_adjectives_corpus.ndjson" 
NUM_PROCESSES = cpu_count() - 1  
CHUNK_SIZE = 10000  

log = logging.getLogger("adjective_analyzer")
log.setLevel(logging.INFO)
handler = logging.StreamHandler()
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
handler.setFormatter(log_formatter)
log.addHandler(handler)

# compiled once globally
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
    
    # adjectives/adverbs by their head
    modifiers_by_head = defaultdict(list)
    
    # children by parent (for traversing tree)
    children_by_parent = defaultdict(list)
    
    # copular verb structures (verb -> subject/predicate)
    copular_structures = defaultdict(lambda: {'subjects': [], 'predicates': []})
    
    # relative clause structures
    relative_clauses = defaultdict(list)
    
    for token in full_tree:
        idx = token.get('idx') or token.get('token_idx')
        head_idx = token.get('head_idx')
        
        # build parent-child relationships
        if head_idx is not None and head_idx != idx:  # avoid self-loops
            children_by_parent[head_idx].append(idx)
        
        # collect adjective/adverb modifiers
        if token.get('dep') in ['amod', 'acomp', 'advmod', 'xcomp'] and token.get('pos') in ['ADJ', 'ADV']:
            modifiers_by_head[head_idx].append({
                'lemma': token.get('lemma', '').lower(),
                'text': token.get('text', ''),
                'dep': token.get('dep'),
                'pos': token.get('pos')
            })
        
        # track copular structures (is/are/was/were/seems/appears)
        lemma_lower = token.get('lemma', '').lower()
        if token.get('pos') == 'AUX' or (token.get('pos') == 'VERB' and lemma_lower in 
                                      ['be', 'seem', 'appear', 'look', 'sound', 'feel', 'become', 'remain']):
            # find subjects
            for child_idx in children_by_parent[idx]:
                child = token_info.get(child_idx)
                if child and child.get('dep') in ['nsubj', 'nsubjpass']:
                    copular_structures[idx]['subjects'].append(child_idx)
            
            # find predicates
            for child_idx in children_by_parent[idx]:
                child = token_info.get(child_idx)
                if child and child.get('dep') in ['acomp', 'attr', 'xcomp']:
                    copular_structures[idx]['predicates'].append(child_idx)
        
        # track relative clauses
        if token.get('dep') in ['relcl', 'acl']:
            relative_clauses[head_idx].append(idx)
    
    return {
        'token_info': token_info,
        'modifiers_by_head': modifiers_by_head,
        'children_by_parent': children_by_parent,
        'copular_structures': copular_structures,
        'relative_clauses': relative_clauses
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


def traverse_for_adjectives(structures, start_idx, max_depth=2):
    adjectives = []
    visited = set()
    
    def traverse(idx, depth):
        if depth > max_depth or idx in visited or idx is None:
            return
        
        visited.add(idx)
        
        # get direct modifiers
        if idx in structures['modifiers_by_head']:
            adjectives.extend(structures['modifiers_by_head'][idx])
        
        # traverse children
        for child_idx in structures['children_by_parent'].get(idx, []):
            if child_idx not in visited:
                traverse(child_idx, depth + 1)
    
    traverse(start_idx, 0)
    return adjectives


def extract_adjectives_comprehensive(structures, model_token_indices):
    adjectives = []
    token_info = structures['token_info']
    
    for token_idx in model_token_indices:
        if token_idx not in token_info:
            continue
        
        token = token_info[token_idx]
    
        
        # PATTERN 1: direct adjectival modification
        # "powerful GPT-4", "the new model"
        if token_idx in structures['modifiers_by_head']:
            adjectives.extend(structures['modifiers_by_head'][token_idx])
        
        # PATTERN 2: model as subject with copular verb
        # "GPT-4 is impressive", "the model seems good"
        if token.get('dep') in ['nsubj', 'nsubjpass']:
            head_idx = token.get('head_idx')
            if head_idx and head_idx in structures['copular_structures']:
                # get predicates of the copular verb
                for pred_idx in structures['copular_structures'][head_idx]['predicates']:
                    # get adjectives modifying the predicate
                    if pred_idx in structures['modifiers_by_head']:
                        adjectives.extend(structures['modifiers_by_head'][pred_idx])
                    # the predicate itself might be an adjective
                    if pred_idx in token_info:
                        pred_token = token_info[pred_idx]
                        if pred_token.get('pos') == 'ADJ':
                            adjectives.append({
                                'lemma': pred_token.get('lemma', '').lower(),
                                'text': pred_token.get('text', ''),
                                'dep': 'acomp',
                                'pos': 'ADJ'
                            })
                
                # also check for adjectives directly modifying the copular verb
                if head_idx in structures['modifiers_by_head']:
                    adjectives.extend(structures['modifiers_by_head'][head_idx])
        
        # PATTERN 3: model as object with adjective complement
        # "I find GPT-4 impressive", "makes the model better"
        if token.get('dep') in ['dobj', 'obj']:
            head_idx = token.get('head_idx')
            if head_idx:
                # look for xcomp (open clausal complement)
                for child_idx in structures['children_by_parent'].get(head_idx, []):
                    child = token_info.get(child_idx)
                    if child and child.get('dep') in ['xcomp', 'acomp']:
                        if child.get('pos') == 'ADJ':
                            adjectives.append({
                                'lemma': child.get('lemma', '').lower(),
                                'text': child.get('text', ''),
                                'dep': 'xcomp',
                                'pos': 'ADJ'
                            })
                        # also get modifiers of the complement
                        if child_idx in structures['modifiers_by_head']:
                            adjectives.extend(structures['modifiers_by_head'][child_idx])
        
        # PATTERN 4: relative clauses
        # "GPT-4, which is powerful, ..."
        if token_idx in structures['relative_clauses']:
            for rel_idx in structures['relative_clauses'][token_idx]:
                # traverse the relative clause for adjectives
                rel_adjectives = traverse_for_adjectives(structures, rel_idx, max_depth=3)
                adjectives.extend(rel_adjectives)
        
        # PATTERN 5: prepositional phrases with adjectives
        # "the performance of GPT-4 is excellent" (captures excellent)
        # this requires looking at what modifies the parent of the model
        head_idx = token.get('head_idx')
        if head_idx and head_idx in token_info:
            parent_token = token_info[head_idx]
            # If model is in a prep phrase, check parent's head
            if token.get('dep') in ['pobj', 'pcomp']:
                grandparent_idx = parent_token.get('head_idx')
                if grandparent_idx and grandparent_idx in structures['modifiers_by_head']:
                    adjectives.extend(structures['modifiers_by_head'][grandparent_idx])
        
        # PATTERN 6: appositive constructions
        # "GPT-4, an impressive model, ..."
        head_idx = token.get('head_idx')
        if head_idx:
            for sibling_idx in structures['children_by_parent'].get(head_idx, []):
                sibling = token_info.get(sibling_idx)
                if sibling and sibling.get('dep') == 'appos':
                    # get adjectives modifying the appositive
                    if sibling_idx in structures['modifiers_by_head']:
                        adjectives.extend(structures['modifiers_by_head'][sibling_idx])
        
        # PATTERN 7: conjunctions
        # "GPT-4 is fast and accurate"
        head_idx = token.get('head_idx')
        if head_idx:
            for child_idx in structures['children_by_parent'].get(head_idx, []):
                child = token_info.get(child_idx)
                if child and child.get('dep') in ['conj', 'cc']:
                    # get adjectives in conjoined clause
                    if child_idx in structures['modifiers_by_head']:
                        adjectives.extend(structures['modifiers_by_head'][child_idx])
        
        # PATTERN 8: comparative constructions
        # "better than GPT-4", "more powerful than the model"
        if token.get('dep') in ['pobj', 'obj']:
            head_idx = token.get('head_idx')
            if head_idx and head_idx in token_info:
                parent = token_info[head_idx]
                # check if parent is "than"
                if parent.get('lemma', '').lower() == 'than':
                    # get the comparative adjective
                    comp_head_idx = parent.get('head_idx')
                    if comp_head_idx and comp_head_idx in token_info:
                        comp_token = token_info[comp_head_idx]
                        if comp_token.get('pos') == 'ADJ':
                            adjectives.append({
                                'lemma': comp_token.get('lemma', '').lower(),
                                'text': comp_token.get('text', ''),
                                'dep': 'comparative',
                                'pos': 'ADJ'
                            })
        
        # PATTERN 9: participles acting as adjectives
        # "GPT-4 has improved", "the updated model"
        for child_idx in structures['children_by_parent'].get(token_idx, []):
            child = token_info.get(child_idx)
            if child and child.get('pos') == 'VERB' and child.get('dep') in ['acl', 'relcl']:
                # this is a participial modifier
                tag = child.get('tag', '')
                if tag in ['VBN', 'VBG']:  # past/present participle
                    adjectives.append({
                        'lemma': child.get('lemma', '').lower(),
                        'text': child.get('text', ''),
                        'dep': 'participial',
                        'pos': 'VERB'
                    })
    
    return adjectives


def process_entry(line):
    try:
        entry = json.loads(line)
        
        # extract the body text based on type
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
        
        # get dependency parse column data
        spacy_parse = entry.get("spacy_parse")
        if not spacy_parse or not spacy_parse.get('sentences'):
            return None
        
        # convert spacy_parse sentences format to flat token list
        full_tree = []
        token_offset = 0
        for sent in spacy_parse['sentences']:
            for token_data in sent['tokens']:
                # adjust indices to be global across all sentences
                token_data['idx'] = token_data['token_idx']
                full_tree.append(token_data)
            token_offset += len(sent['tokens'])
        
        # build dependency structures
        structures = build_dependency_structures(full_tree)
        
        # find model tokens
        model_tokens = find_model_token_indices_enhanced(full_tree, text_lower, models_found)
        
        results = []
        
        for model_name in models_found:
            token_indices = model_tokens.get(model_name, [])
            if not token_indices:
                continue
            
            # extract adjectives
            adjectives = extract_adjectives_comprehensive(structures, token_indices)
            
            if adjectives:
                output_entry = {
                    "id": entry.get("id"),
                    "model_detected": model_name,
                    "created_date": entry.get("created_date"),
                    "text": text,
                    "adjectives": [adj['lemma'] for adj in adjectives]  # just the lemmas
                }
                results.append(output_entry)
        
        return results if results else None
        
    except Exception:
        return None


def process_chunk(lines):
    chunk_results = []
    entries_processed = 0
    
    for line in lines:
        result = process_entry(line)
        if result:
            entries_processed += len(result)
            chunk_results.extend(result)
    
    return chunk_results, entries_processed


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
    log.info(f"using {NUM_PROCESSES} processes")
    log.info("counting entries...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    log.info(f"total entries: {total_lines:,}")
    
    total_processed = 0
    
    log.info("processing entries in parallel...")
    
    # open output file for writing
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        with Pool(processes=NUM_PROCESSES) as pool:
            # read file in chunks and process in parallel
            chunks = list(read_in_chunks(INPUT_FILE, CHUNK_SIZE))
            with tqdm(total=len(chunks), desc="processing chunks", unit="chunk") as pbar:
                for chunk_results, entries_in_chunk in pool.imap_unordered(process_chunk, chunks):
                    # write each result as a line in the NDJSON file
                    for result in chunk_results:
                        outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
                    
                    total_processed += entries_in_chunk
                    pbar.update(1)
                    pbar.set_postfix({'entries_with_adjectives': total_processed})
    
    log.info(f"\nprocessed {total_processed:,} entries with adjectives")
    log.info(f"\n ^_^ results saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()