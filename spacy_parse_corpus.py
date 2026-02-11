import spacy
import json
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import time

# entry["spacy_parse"]

INPUT_FILE = "mini_corpus_with_scores.ndjson"  # file with JSON files
OUTPUT_FILE = "mini_corpus_with_scores_deps.ndjson"  # output file for parsed files
BATCH_SIZE = 32  # smaller batches for transformer models (memory intensive)
PROGRESS_INTERVAL = 5000  # log progress every N documents

log = logging.getLogger("reddit_spacy_parser")
log.setLevel(logging.INFO)
handler = logging.StreamHandler()
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
handler.setFormatter(log_formatter)
log.addHandler(handler)

# Load transformer model
nlp = spacy.load("en_core_web_trf")

def extract_dependency_parse(doc):
    sentences = []
    for sent in doc.sents:
        tokens = []
        for token in sent:
            token_data = {
                "text": token.text,
                "lemma": token.lemma_,
                "pos": token.pos_,  # universal POS tag
                "tag": token.tag_,  # fine-grained POS tag
                "dep": token.dep_,  # dependency relation
                "head_text": token.head.text,
                "head_pos": token.head.pos_,
                "head_idx": token.head.i,
                "token_idx": token.i,
                "is_stop": token.is_stop,
                "is_alpha": token.is_alpha,
                "is_punct": token.is_punct,
                "children": [child.text for child in token.children],
                "subtree": [t.text for t in token.subtree][:10]  # limit subtree size
            }
            tokens.append(token_data)
        
        sentence_data = {
            "text": sent.text,
            "tokens": tokens,
            "root": sent.root.text,
            "root_dep": sent.root.dep_,
            "root_pos": sent.root.pos_
        }
        sentences.append(sentence_data)
    
    return {
        "num_sentences": len(sentences),
        "num_tokens": len(doc),
        "sentences": sentences
    }


def get_text_to_parse(entry):
    if entry.get("type") == "submission":
        title = entry.get("title", "")
        selftext = entry.get("selftext", "")
        return f"{title} {selftext}".strip()
    else:  # comment
        return entry.get("body", "").strip()


def process_file(nlp, input_file, output_file):
    log.info(f"Processing: {input_file}")
    
    # count total entries
    total_entries = 0
    with open(input_file, 'r', encoding='utf-8') as f:
        for _ in f:
            total_entries += 1
    
    log.info(f"  total entries to process: {total_entries:,}")
    
    processed = 0
    skipped = 0
    batch = []
    batch_texts = []
    batch_entries = []
    
    start_time = time.time()
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                entry = json.loads(line)
                text = get_text_to_parse(entry)
                
                if not text or len(text.strip()) == 0:
                    skipped += 1
                    continue
                
                # truncate to avoid memory issues - more aggressive for trf
                # transformers have max sequence length (typically 512 tokens)
                if len(text) > 5000:  # reduced from 10000
                    text = text[:5000]
                
                batch_texts.append(text)
                batch_entries.append(entry)
                
                if len(batch_texts) >= BATCH_SIZE:
                    process_batch(nlp, batch_entries, batch_texts, outfile)
                    processed += len(batch_texts)
                    
                    # log progress
                    if processed % PROGRESS_INTERVAL == 0 or processed == total_entries:
                        elapsed = time.time() - start_time
                        rate = processed / elapsed if elapsed > 0 else 0
                        remaining = total_entries - processed
                        eta_seconds = remaining / rate if rate > 0 else 0
                        eta_str = format_time(eta_seconds)
                        
                        log.info(
                            f"  progress: {processed:,}/{total_entries:,} "
                            f"({100*processed/total_entries:.1f}%) | "
                            f"rate: {rate:.1f} docs/sec | "
                            f"elapsed: {format_time(elapsed)} | "
                            f"ETA: {eta_str}"
                        )
                    
                    batch_texts = []
                    batch_entries = []
            
            except json.JSONDecodeError:
                skipped += 1
                continue
            except Exception as e:
                log.warning(f"  error on line {line_num}: {e}")
                skipped += 1
                continue
        
        if batch_texts:
            process_batch(nlp, batch_entries, batch_texts, outfile)
            processed += len(batch_texts)
    
    elapsed = time.time() - start_time
    rate = processed / elapsed if elapsed > 0 else 0
    
    log.info(f"  complete: {input_file}")
    log.info(f"  processed: {processed:,} | skipped: {skipped:,} | time: {format_time(elapsed)} | Rate: {rate:.1f} docs/sec")
    
    return processed, skipped


def process_batch(nlp, entries, texts, outfile):
    # smaller internal batch size for transformer memory efficiency
    docs = list(nlp.pipe(texts, batch_size=16))  # reduced from 50
    for entry, doc in zip(entries, docs):
        try:
            # add dependency parse data
            entry["spacy_parse"] = extract_dependency_parse(doc)
            
            # write to output
            outfile.write(json.dumps(entry) + '\n')
        except Exception as e:
            log.warning(f"  error processing entry {entry.get('id')}: {e}")
            continue


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        mins = seconds / 60
        return f"{mins:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def estimate_total_time(nlp, sample_file):
    sample_texts = []
    with open(sample_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 100:  # sample first 100 entries
                break
            try:
                entry = json.loads(line)
                text = get_text_to_parse(entry)
                if text:
                    sample_texts.append(text[:5000])  # reduced from 10000
            except:
                continue
    
    if not sample_texts:
        log.warning("could not read sample texts")
        return None
    
    start = time.time()
    list(nlp.pipe(sample_texts, batch_size=16))  # reduced from 50
    elapsed = time.time() - start
    
    rate = len(sample_texts) / elapsed
    log.info(f"  sample rate: {rate:.1f} docs/sec")
    
    return rate


def main():
    if not Path(INPUT_FILE).exists():
        log.error(f"input file not found: {INPUT_FILE}")
        return
    
    estimate_total_time(nlp, INPUT_FILE)
    log.info("")
    
    overall_start = time.time()
    
    try:
        total_processed, total_skipped = process_file(nlp, INPUT_FILE, OUTPUT_FILE)
    except Exception as e:
        log.error(f"error processing {INPUT_FILE}: {e}")
        return
    
    overall_elapsed = time.time() - overall_start
    overall_rate = total_processed / overall_elapsed if overall_elapsed > 0 else 0
    
    log.info(f"total entries processed: {total_processed:,}")
    log.info(f"total entries skipped: {total_skipped:,}")
    log.info(f"total time: {format_time(overall_elapsed)}")
    log.info(f"overall rate: {overall_rate:.1f} docs/sec")
    log.info(f"output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()