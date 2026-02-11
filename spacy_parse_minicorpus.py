import spacy
import json
from pathlib import Path
import logging
from tqdm import tqdm
from typing import Dict, List
import sys

# entry["dependency_parse"], fix this different column name

INPUT_FILE = "mini_corpus_with_scores.ndjson"
OUTPUT_FILE = "mini_corpus_with_scores_deps.ndjson"
BATCH_SIZE = 100  # for efficiency
MAX_LENGTH = 1000000  # spaCy default max length

log = logging.getLogger("spacy_parser")
log.setLevel(logging.INFO)
handler = logging.StreamHandler()
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
handler.setFormatter(log_formatter)
log.addHandler(handler)


def serialize_dependency_tree(doc) -> List[Dict]:
    tree = []
    for token in doc:
        token_info = {
            "text": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,  
            "tag": token.tag_,  # fine-grained POS tag
            "dep": token.dep_,  # dependency relation
            "head_text": token.head.text,  # text of the head token
            "head_idx": token.head.i,  # index of the head token
            "idx": token.i,  # token's own index
            "is_root": token.dep_ == "ROOT",
            "children": [child.i for child in token.children]  # indices of children
        }
        tree.append(token_info)
    return tree


def get_dependency_structure_compact(doc) -> Dict:
    triples = []
    for token in doc:
        if token.dep_ != "ROOT":
            triples.append({
                "head": token.head.i,
                "head_text": token.head.text,
                "rel": token.dep_,
                "dep": token.i,
                "dep_text": token.text
            })
    
    # store sentence boundaries
    sentences = [{"start": sent.start, "end": sent.end} for sent in doc.sents]
    
    return {
        "triples": triples,
        "sentences": sentences,
        "root_indices": [token.i for token in doc if token.dep_ == "ROOT"]
    }


def process_batch(texts: List[str], nlp) -> List[Dict]:
    results = []
    
    for doc in nlp.pipe(texts, batch_size=50, disable=["ner"]):  # disable NER for speed
        parsed_data = {
            "full_tree": serialize_dependency_tree(doc),
            "compact": get_dependency_structure_compact(doc),
            "num_tokens": len(doc),
            "num_sentences": len(list(doc.sents))
        }
        results.append(parsed_data)
    
    return results


def count_lines(file_path: str) -> int:
    log.info("counting lines in input file...")
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)


def main():
    log.info("loading spaCy model (this may take a moment)...")
    
    # Load English model - using 'en_core_web_sm' for speed
    # For better accuracy, use 'en_core_web_trf' (transformer-based, but much slower)
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        log.error("spaCy model not found. please run: python -m spacy download en_core_web_sm")
        sys.exit(1)
    
    nlp.max_length = MAX_LENGTH
    
    log.info(f"model loaded: {nlp.meta['name']} (version {nlp.meta['version']})")
    log.info(f"processing file: {INPUT_FILE}")
    
    total_lines = count_lines(INPUT_FILE)
    log.info(f"total entries to process: {total_lines:,}")
    
    batch = []
    batch_entries = []
    processed = 0
    errors = 0
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as infile, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        
        with tqdm(total=total_lines, desc="processing", unit="entries") as pbar:
            for line in infile:
                try:
                    entry = json.loads(line)
                    
                    if entry.get("type") == "submission":
                        # combine title and selftext for submissions
                        text = f"{entry.get('title', '')} {entry.get('selftext', '')}".strip()
                    else:  # comment
                        text = entry.get("body", "").strip()
                    
                    if not text:
                        # skip empty entries
                        entry["dependency_parse"] = None
                        outfile.write(json.dumps(entry) + '\n')
                        processed += 1
                        pbar.update(1)
                        continue
                    
                    batch.append(text)
                    batch_entries.append(entry)
                    
                    if len(batch) >= BATCH_SIZE:
                        parsed_results = process_batch(batch, nlp)
                        
                        for entry, parsed_data in zip(batch_entries, parsed_results):
                            entry["dependency_parse"] = parsed_data
                            outfile.write(json.dumps(entry) + '\n')
                        
                        processed += len(batch)
                        pbar.update(len(batch))
                        
                        batch = []
                        batch_entries = []
                    
                except json.JSONDecodeError as e:
                    log.warning(f"JSON decode error on line {processed + 1}: {e}")
                    errors += 1
                    pbar.update(1)
                except Exception as e:
                    log.warning(f"error processing entry {processed + 1}: {e}")
                    errors += 1
                    pbar.update(1)
            
            if batch:
                try:
                    parsed_results = process_batch(batch, nlp)
                    for entry, parsed_data in zip(batch_entries, parsed_results):
                        entry["dependency_parse"] = parsed_data
                        outfile.write(json.dumps(entry) + '\n')
                    processed += len(batch)
                    pbar.update(len(batch))
                except Exception as e:
                    log.error(f"error processing final batch: {e}")
                    errors += len(batch)
    
    log.info("processing complete!")
    log.info(f"total processed: {processed:,}")
    log.info(f"errors: {errors:,}")
    log.info(f"output saved to: {OUTPUT_FILE}")
    
    # show example of output structure
    log.info("\nexample dependency parse structure:")
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        first_entry = json.loads(f.readline())
        if first_entry.get("dependency_parse"):
            log.info(f"  - full tree tokens: {first_entry['dependency_parse']['num_tokens']}")
            log.info(f"  - sentences: {first_entry['dependency_parse']['num_sentences']}")
            log.info(f"  - dependency triples: {len(first_entry['dependency_parse']['compact']['triples'])}")


if __name__ == "__main__":
    main()