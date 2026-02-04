import os
import sys
from pathlib import Path
from typing import List

def get_folders(base_path: str) -> List[Path]:
    base = Path(base_path)
    
    # generate folder names in order
    folders = []
    
    # pushshift_25_12_json through pushshift_07_12_json (descending days in December)
    for day in range(25, 6, -1):
        folder_name = f"pushshift_25_{day:02d}_json"
        folder_path = base / folder_name
        if folder_path.exists() and folder_path.is_dir():
            folders.append(folder_path)
        else:
            print(f"folder not found: {folder_path}", file=sys.stderr)
    
    # pushshift_25_01_06_json (specific date)
    folder_path = base / "pushshift_25_01_06_json"
    if folder_path.exists() and folder_path.is_dir():
        folders.append(folder_path)
    else:
        print(f"folder not found: {folder_path}", file=sys.stderr)
    
    # pushshift_22_24_json
    folder_path = base / "pushshift_22_24_json"
    if folder_path.exists() and folder_path.is_dir():
        folders.append(folder_path)
    else:
        print(f"folder not found: {folder_path}", file=sys.stderr)
    
    return folders

def get_ndjson_files(folder: Path) -> List[Path]:
    
    files = []
    for ext in ['*.json', '*.ndjson', '*.jsonl']:
        files.extend(folder.glob(ext))
    return sorted(files)

def combine_files(folders: List[Path], output_file: str, verbose: bool = True):
    total_lines = 0
    total_files = 0
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for folder in folders:
            if verbose:
                print(f"processing folder: {folder.name}")
            
            files = get_ndjson_files(folder)
            
            if not files:
                print(f"  no JSON files found in {folder.name}", file=sys.stderr)
                continue
            
            for json_file in files:
                if verbose:
                    print(f"  reading: {json_file.name}")
                
                try:
                    with open(json_file, 'r', encoding='utf-8') as infile:
                        file_lines = 0
                        for line in infile:
                            line = line.strip()
                            if line:  
                                outfile.write(line + '\n')
                                file_lines += 1
                                total_lines += 1
                    
                    if verbose:
                        print(f"    added {file_lines:,} lines")
                    total_files += 1
                    
                except Exception as e:
                    print(f"  error reading {json_file}: {e}", file=sys.stderr)
    
    if verbose:
        print(f"\ncombined {total_files} files into {output_file}")
        print(f"total lines: {total_lines:,}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='combine newline-delimited JSON files from multiple folders'
    )
    parser.add_argument(
        'base_path',
        help='base directory containing the pushshift folders'
    )
    parser.add_argument(
        '-o', '--output',
        default='combined_corpus.ndjson',
        help='output file name (default: combined_corpus.ndjson)'
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='suppress progress output'
    )
    
    args = parser.parse_args()
    
    folders = get_folders(args.base_path)
    
    if not folders:
        print("error: no valid folders found!", file=sys.stderr)
        sys.exit(1)
    
    if not args.quiet:
        print(f"found {len(folders)} folders to process")
        print(f"output file: {args.output}\n")
    
    # combine files
    combine_files(folders, args.output, verbose=not args.quiet)
    
    if not args.quiet:
        print(f"\ndone ^_^ output written to: {args.output}")

if __name__ == '__main__':
    main()