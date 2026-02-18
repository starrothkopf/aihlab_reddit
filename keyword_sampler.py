import json
import random
import argparse
from pathlib import Path
from collections import defaultdict
import logging
import re

log = logging.getLogger("keyword_sampler")
log.setLevel(logging.INFO)
handler = logging.StreamHandler()
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
handler.setFormatter(log_formatter)
log.addHandler(handler)


def extract_context(text, keyword, context_chars=200, whole_word=True):
    """
    extract text around the keyword occurrence with surrounding context.
    
    args:
        text: full text to search in
        keyword: the keyword to find
        context_chars: number of characters to include before/after keyword
        whole_word: whether to match whole words only
    
    returns:
        list of context snippets (can be multiple if keyword appears multiple times)
    """
    contexts = []
    
    if whole_word:
        # use regex with word boundaries
        escaped_keyword = re.escape(keyword)
        pattern = r'\b' + escaped_keyword + r'\b'
        matches = re.finditer(pattern, text, re.IGNORECASE)
        
        for match in matches:
            idx = match.start()
            
            # get context window
            context_start = max(0, idx - context_chars)
            context_end = min(len(text), idx + len(match.group(0)) + context_chars)
            
            # extract context and add markers
            snippet = text[context_start:context_end]
            
            # add ellipsis if we're not at the beginning/end
            if context_start > 0:
                snippet = "..." + snippet
            if context_end < len(text):
                snippet = snippet + "..."
            
            contexts.append(snippet)
    else:
        # original simple substring search
        text_lower = text.lower()
        keyword_lower = keyword.lower()
        
        start = 0
        while True:
            idx = text_lower.find(keyword_lower, start)
            if idx == -1:
                break
            
            # get context window
            context_start = max(0, idx - context_chars)
            context_end = min(len(text), idx + len(keyword) + context_chars)
            
            # extract context and add markers
            snippet = text[context_start:context_end]
            
            # add ellipsis if we're not at the beginning/end
            if context_start > 0:
                snippet = "..." + snippet
            if context_end < len(text):
                snippet = snippet + "..."
            
            contexts.append(snippet)
            start = idx + 1
    
    return contexts


def search_corpus(input_file, keyword, sample_size=100, case_sensitive=False, whole_word=True):
    """
    search through the corpus and sample random entries containing the keyword.
    
    args:
        input_file: path to the ndjson corpus file
        keyword: the keyword to search for
        sample_size: maximum number of samples to return
        case_sensitive: whether to do case-sensitive matching
        whole_word: whether to match whole words only (avoid partial matches like "both" when searching "bot")
    
    returns:
        list of sampled entries with context
    """
    log.info(f"searching for '{keyword}' in {input_file}")
    log.info(f"case sensitive: {case_sensitive}")
    log.info(f"whole word matching: {whole_word}")
    
    matching_entries = []
    total_processed = 0
    total_matches = 0
    stats = {
        "by_type": defaultdict(int),
        "by_year": defaultdict(int),
        "by_subreddit": defaultdict(int),
        "by_model": defaultdict(int)
    }
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            total_processed += 1
            
            if total_processed % 100000 == 0:
                log.info(f"processed {total_processed:,} entries, found {total_matches:,} matches")
            
            try:
                entry = json.loads(line)
                
                # get searchable text based on entry type
                if entry.get("type") == "submission":
                    searchable_text = (entry.get("title", "") + " " + 
                                     entry.get("selftext", ""))
                else:  # comment
                    searchable_text = entry.get("body", "")
                
                # check if keyword exists
                if whole_word:
                    # use word boundaries to match whole words only
                    # escape special regex characters in the keyword
                    escaped_keyword = re.escape(keyword)
                    pattern = r'\b' + escaped_keyword + r'\b'
                    flags = re.IGNORECASE if not case_sensitive else 0
                    contains_keyword = bool(re.search(pattern, searchable_text, flags))
                else:
                    # simple substring matching
                    if case_sensitive:
                        contains_keyword = keyword in searchable_text
                    else:
                        contains_keyword = keyword.lower() in searchable_text.lower()
                
                if contains_keyword:
                    total_matches += 1
                    
                    # extract context around keyword
                    contexts = extract_context(searchable_text, keyword, whole_word=whole_word)
                    
                    # add context to entry
                    entry["keyword_contexts"] = contexts
                    entry["keyword_occurrences"] = len(contexts)
                    
                    matching_entries.append(entry)
                    
                    # update stats
                    stats["by_type"][entry.get("type", "unknown")] += 1
                    
                    # extract year from created_date
                    created_date = entry.get("created_date", "")
                    if created_date:
                        year = created_date.split("-")[0]
                        stats["by_year"][year] += 1
                    
                    stats["by_subreddit"][entry.get("subreddit", "unknown")] += 1
                    stats["by_model"][entry.get("model_detected", "unknown")] += 1
                    
            except json.JSONDecodeError:
                continue
    
    log.info(f"total entries processed: {total_processed:,}")
    log.info(f"total matches found: {total_matches:,}")
    
    # sample if we have more matches than requested
    if len(matching_entries) > sample_size:
        sampled = random.sample(matching_entries, sample_size)
        log.info(f"randomly sampled {sample_size} entries from {len(matching_entries):,} matches")
    else:
        sampled = matching_entries
        log.info(f"returning all {len(matching_entries)} matches (less than sample size of {sample_size})")
    
    return sampled, stats, total_matches


def print_sample_contexts(samples, keyword, max_display=10):
    
    for i, entry in enumerate(samples[:max_display], 1):
        print(f"\n[{i}] {entry['type'].upper()} in r/{entry.get('subreddit', 'unknown')}")
        print(f"    date: {entry.get('created_date', 'unknown')}")
        print(f"    author: {entry.get('author', 'unknown')}")
        print(f"    model detected: {entry.get('model_detected', 'unknown')}")
        print(f"    keyword occurrences: {entry.get('keyword_occurrences', 0)}")
        
        if entry.get('full_link'):
            print(f"    Link: {entry['full_link']}")
        
        print(f"\n    context(s):")
        for j, context in enumerate(entry.get('keyword_contexts', []), 1):
            print(f"      [{j}] {context}")
        print("-" * 80)
    
    if len(samples) > max_display:
        print(f"\n... and {len(samples) - max_display} more entries")


def print_statistics(stats, total_matches):
    print(f"\ntotal matches: {total_matches:,}")
    
    print("\nby type:")
    for type_name, count in sorted(stats["by_type"].items(), key=lambda x: -x[1]):
        print(f"  {type_name}: {count:,}")
    
    print("\nby year:")
    for year, count in sorted(stats["by_year"].items()):
        print(f"  {year}: {count:,}")
    
    print("\nby model detected:")
    for model, count in sorted(stats["by_model"].items(), key=lambda x: -x[1]):
        print(f"  {model}: {count:,}")


def save_results(samples, output_file, keyword):
    output_data = {
        "keyword": keyword,
        "sample_size": len(samples),
        "samples": samples
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    log.info(f"\nresults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="sample random mentions of a keyword from the Reddit corpus"
    )
    parser.add_argument(
        "keyword",
        help="the keyword to search for (e.g., 'bot', 'AI', 'human')"
    )
    parser.add_argument(
        "-i", "--input",
        default="combined_corpus.ndjson",
        help="input corpus file (default: combined_corpus.ndjson)"
    )
    parser.add_argument(
        "-n", "--sample-size",
        type=int,
        default=100,
        help="number of random samples to return (default: 100)"
    )
    parser.add_argument(
        "-o", "--output",
        help="output JSON file (default: keyword_sample_{keyword}.json)"
    )
    parser.add_argument(
        "-c", "--case-sensitive",
        action="store_true",
        help="enable case-sensitive search"
    )
    parser.add_argument(
        "-p", "--partial-match",
        action="store_true",
        help="allow partial word matches (e.g., 'bot' will match 'both', 'robot'). default is whole-word matching only."
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="skip printing preview of samples to console"
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=10,
        help="number of samples to preview in console (default: 10)"
    )
    
    args = parser.parse_args()
    
    # set output filename
    if args.output is None:
        safe_keyword = "".join(c if c.isalnum() else "_" for c in args.keyword)
        args.output = f"keyword_sample_{safe_keyword}.json"
    
    # search and sample
    samples, stats, total_matches = search_corpus(
        args.input,
        args.keyword,
        args.sample_size,
        args.case_sensitive,
        whole_word=not args.partial_match  # Invert: default is whole word, flag enables partial
    )
    
    if not samples:
        log.warning(f"No matches found for '{args.keyword}'")
        return
    
    print_statistics(stats, total_matches)
    
    if not args.no_preview:
        print_sample_contexts(samples, args.keyword, args.preview_count)
    
    save_results(samples, args.output, args.keyword)
    
    print(f"^_^ successfully sampled {len(samples)} mentions of '{args.keyword}'")



if __name__ == "__main__":
    main()