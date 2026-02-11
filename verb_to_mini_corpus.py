import json
import re
import csv
from collections import defaultdict, Counter
from pathlib import Path
import logging

INPUT_FILE = "mini_corpus_with_scores_deps.ndjson" 
OUTPUT_FILE = "verbs_scores_inspection.csv"

log = logging.getLogger("verb_analyzer")
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
    
    # verbs by their subjects
    verbs_by_subject = defaultdict(list)
    
    # verbs by their objects
    verbs_by_object = defaultdict(list)
    
    # passive constructions
    passive_constructions = defaultdict(dict)
    
    for token in full_tree:
        idx = token.get('idx') or token.get('token_idx')
        head_idx = token.get('head_idx')
        
        # build parent-child relationships
        if head_idx is not None and head_idx != idx:
            children_by_parent[head_idx].append(idx)
        
        # track verb-subject relationships
        if token.get('dep') in ['nsubj', 'nsubjpass']:
            if head_idx in token_info:
                head = token_info[head_idx]
                if head.get('pos') in ['VERB', 'AUX']:
                    verbs_by_subject[idx].append({
                        'verb_idx': head_idx,
                        'lemma': head.get('lemma', '').lower(),
                        'text': head.get('text', ''),
                        'is_passive': token.get('dep') == 'nsubjpass'
                    })
        
        # track verb-object relationships
        if token.get('dep') in ['dobj', 'obj', 'iobj', 'pobj']:
            if head_idx in token_info:
                head = token_info[head_idx]
                if head.get('pos') in ['VERB', 'AUX']:
                    verbs_by_object[idx].append({
                        'verb_idx': head_idx,
                        'lemma': head.get('lemma', '').lower(),
                        'text': head.get('text', ''),
                        'dep': token.get('dep')
                    })
        
        # identify passive constructions (auxiliary + past participle)
        if token.get('pos') == 'AUX' and token.get('lemma', '').lower() in ['be', 'get']:
            # look for past participle children
            for child_idx in children_by_parent.get(idx, []):
                child = token_info.get(child_idx)
                if child and child.get('tag') == 'VBN':  # past participle
                    passive_constructions[child_idx] = {
                        'aux_idx': idx,
                        'aux_lemma': token.get('lemma', '').lower()
                    }
    
    return {
        'token_info': token_info,
        'children_by_parent': children_by_parent,
        'verbs_by_subject': verbs_by_subject,
        'verbs_by_object': verbs_by_object,
        'passive_constructions': passive_constructions
    }


def find_model_token_indices(full_tree, text_lower, model_names):
    model_tokens = defaultdict(set)
    
    # build a map of text positions to token indices
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


def extract_verbs_comprehensive(structures, model_token_indices):
    verbs_by_model = []  # model is the agent (subject)
    verbs_to_model = []  # model is the patient (object/passive subject)
    
    token_info = structures['token_info']
    
    for token_idx in model_token_indices:
        if token_idx not in token_info:
            continue
        
        token = token_info[token_idx]
        
        # PATTERN 1: model as active subject
        # "GPT-4 generates text", "the model understands context"
        if token_idx in structures['verbs_by_subject']:
            for verb_info in structures['verbs_by_subject'][token_idx]:
                if not verb_info['is_passive']:
                    verbs_by_model.append({
                        'lemma': verb_info['lemma'],
                        'text': verb_info['text'],
                        'pattern': 'active_subject',
                        'example_structure': 'MODEL verb'
                    })
        
        # PATTERN 2: model as passive subject
        # "GPT-4 was trained on data", "the model is being used"
        if token_idx in structures['verbs_by_subject']:
            for verb_info in structures['verbs_by_subject'][token_idx]:
                if verb_info['is_passive']:
                    verb_idx = verb_info['verb_idx']
                    # check if this is part of a passive construction
                    if verb_idx in structures['passive_constructions']:
                        verbs_to_model.append({
                            'lemma': verb_info['lemma'],
                            'text': verb_info['text'],
                            'pattern': 'passive_subject',
                            'example_structure': 'MODEL was verbed'
                        })
                    else:
                        verbs_to_model.append({
                            'lemma': verb_info['lemma'],
                            'text': verb_info['text'],
                            'pattern': 'passive_nsubjpass',
                            'example_structure': 'MODEL is verbed'
                        })
        
        # PATTERN 3: model as direct object
        # "I use GPT-4", "people trust the model"
        if token_idx in structures['verbs_by_object']:
            for verb_info in structures['verbs_by_object'][token_idx]:
                if verb_info['dep'] in ['dobj', 'obj']:
                    verbs_to_model.append({
                        'lemma': verb_info['lemma'],
                        'text': verb_info['text'],
                        'pattern': 'direct_object',
                        'example_structure': 'verb MODEL'
                    })
        
        # PATTERN 4: model as prepositional object
        # "rely on GPT-4", "work with the model"
        if token_idx in structures['verbs_by_object']:
            for verb_info in structures['verbs_by_object'][token_idx]:
                if verb_info['dep'] == 'pobj':
                    # get the preposition
                    verb_idx = verb_info['verb_idx']
                    if verb_idx in token_info:
                        # find the preposition child
                        prep = None
                        for child_idx in structures['children_by_parent'].get(verb_idx, []):
                            child = token_info.get(child_idx)
                            if child and child.get('pos') == 'ADP':
                                prep = child.get('text', '')
                                break
                        
                        verbs_to_model.append({
                            'lemma': verb_info['lemma'],
                            'text': verb_info['text'],
                            'pattern': 'prepositional_object',
                            'example_structure': f'verb {prep} MODEL' if prep else 'verb prep MODEL',
                            'preposition': prep
                        })
        
        # PATTERN 5: infinitive constructions where model is subject
        # "GPT-4 can help", "the model will improve"
        head_idx = token.get('head_idx')
        if head_idx and head_idx in token_info:
            head = token_info[head_idx]
            # check for auxiliary verbs
            if head.get('pos') == 'AUX' and head.get('lemma', '').lower() in ['can', 'could', 'will', 'would', 'should', 'may', 'might', 'must']:
                # find the main verb
                for child_idx in structures['children_by_parent'].get(head_idx, []):
                    child = token_info.get(child_idx)
                    if child and child.get('pos') == 'VERB' and child.get('dep') in ['xcomp', 'ccomp']:
                        verbs_by_model.append({
                            'lemma': child.get('lemma', '').lower(),
                            'text': child.get('text', ''),
                            'pattern': 'modal_auxiliary',
                            'example_structure': f'MODEL {head.get("text")} verb'
                        })
        
        # PATTERN 6: verb complements where model is involved
        # "makes GPT-4 useful", "helps the model learn"
        if token.get('dep') in ['dobj', 'obj']:
            head_idx = token.get('head_idx')
            if head_idx and head_idx in token_info:
                # look for xcomp (open clausal complement)
                for child_idx in structures['children_by_parent'].get(head_idx, []):
                    child = token_info.get(child_idx)
                    if child and child.get('dep') == 'xcomp' and child.get('pos') == 'VERB':
                        # this verb describes what the model does
                        verbs_by_model.append({
                            'lemma': child.get('lemma', '').lower(),
                            'text': child.get('text', ''),
                            'pattern': 'object_complement',
                            'example_structure': 'verb MODEL to verb2'
                        })
        
        # PATTERN 7: relative clauses with model as subject
        # "GPT-4, which processes text, ..."
        for child_idx in structures['children_by_parent'].get(token_idx, []):
            child = token_info.get(child_idx)
            if child and child.get('dep') in ['relcl', 'acl']:
                # check if child is a verb
                if child.get('pos') in ['VERB', 'AUX']:
                    # check if the model is the subject of this verb
                    has_explicit_subject = False
                    for grandchild_idx in structures['children_by_parent'].get(child_idx, []):
                        grandchild = token_info.get(grandchild_idx)
                        if grandchild and grandchild.get('dep') in ['nsubj', 'nsubjpass']:
                            has_explicit_subject = True
                            break
                    
                    # if no explicit subject, model is the implicit subject
                    if not has_explicit_subject:
                        verbs_by_model.append({
                            'lemma': child.get('lemma', '').lower(),
                            'text': child.get('text', ''),
                            'pattern': 'relative_clause',
                            'example_structure': 'MODEL which verb'
                        })
        
        # PATTERN 8: conjunctions (capture conjoined verbs)
        # "GPT-4 understands and generates text"
        if token_idx in structures['verbs_by_subject']:
            for verb_info in structures['verbs_by_subject'][token_idx]:
                if not verb_info['is_passive']:
                    verb_idx = verb_info['verb_idx']
                    # look for conjoined verbs
                    for child_idx in structures['children_by_parent'].get(verb_idx, []):
                        child = token_info.get(child_idx)
                        if child and child.get('dep') == 'conj' and child.get('pos') == 'VERB':
                            verbs_by_model.append({
                                'lemma': child.get('lemma', '').lower(),
                                'text': child.get('text', ''),
                                'pattern': 'conjunction',
                                'example_structure': 'MODEL verb1 and verb2'
                            })
    
    return {
        'verbs_by_model': verbs_by_model,
        'verbs_to_model': verbs_to_model
    }


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
        model_tokens = find_model_token_indices(full_tree, text_lower, models_found)
        
        created_date = entry.get('created_date', '')
        is_anthro1 = entry.get('IsAnthro1', '')
        is_anthro2 = entry.get('IsAnthro2', '')
        
        results = []
        
        for model_name in models_found:
            token_indices = model_tokens.get(model_name, [])
            if not token_indices:
                continue
            
            # extract verbs
            verbs = extract_verbs_comprehensive(structures, token_indices)
            
            # add verbs BY model (model does action)
            for verb in verbs['verbs_by_model']:
                results.append({
                    'entry_id': entry_id,
                    'entry_type': entry_type,
                    'created_date': created_date,
                    'text': text[:1000],  # truncate long texts
                    'model': model_name,
                    'verb_lemma': verb['lemma'],
                    'verb_text': verb['text'],
                    'verb_direction': 'by_model',
                    'pattern': verb.get('pattern', 'unknown'),
                    'example_structure': verb.get('example_structure', ''),
                    'IsAnthro1': is_anthro1,
                    'IsAnthro2': is_anthro2
                })
            
            # add verbs TO model (action done to model)
            for verb in verbs['verbs_to_model']:
                results.append({
                    'entry_id': entry_id,
                    'entry_type': entry_type,
                    'created_date': created_date,
                    'text': text[:1000],  
                    'model': model_name,
                    'verb_lemma': verb['lemma'],
                    'verb_text': verb['text'],
                    'verb_direction': 'to_model',
                    'pattern': verb.get('pattern', 'unknown'),
                    'example_structure': verb.get('example_structure', ''),
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
    entries_with_verbs = 0
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for entry_id, line in enumerate(f, 1):
            try:
                entry = json.loads(line)
                entries_processed += 1
                
                results = process_entry(entry, entry_id)
                if results:
                    entries_with_verbs += 1
                    csv_rows.extend(results)
                    
                if entries_processed % 100 == 0:
                    log.info(f"processed {entries_processed} entries, found {entries_with_verbs} with verbs")
                    
            except json.JSONDecodeError:
                log.error(f"JSON decode error on line {entry_id}")
                continue
    
    log.info(f"\ntotal processed: {entries_processed}")
    log.info(f"entries with verbs: {entries_with_verbs}")
    log.info(f"total verb extractions: {len(csv_rows)}")
    
    # write to CSV
    if csv_rows:
        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['entry_id', 'entry_type', 'created_date', 'model', 'verb_lemma', 
                         'verb_text', 'verb_direction', 'pattern', 'example_structure',
                         'text', 'IsAnthro1', 'IsAnthro2']
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
        
        # stats by direction
        by_model_count = sum(1 for row in csv_rows if row['verb_direction'] == 'by_model')
        to_model_count = sum(1 for row in csv_rows if row['verb_direction'] == 'to_model')
        log.info(f"\nverbs BY model (model does action): {by_model_count}")
        log.info(f"verbs TO model (action done to model): {to_model_count}")
        
        verbs_by = Counter(row['verb_lemma'] for row in csv_rows if row['verb_direction'] == 'by_model')
        log.info("\ntop 20 verbs BY model:")
        for verb, count in verbs_by.most_common(20):
            log.info(f"  {verb}: {count}")
        
        verbs_to = Counter(row['verb_lemma'] for row in csv_rows if row['verb_direction'] == 'to_model')
        log.info("\ntop 20 verbs TO model:")
        for verb, count in verbs_to.most_common(20):
            log.info(f"  {verb}: {count}")
    else:
        log.info("no verbs found!")


if __name__ == "__main__":
    main()