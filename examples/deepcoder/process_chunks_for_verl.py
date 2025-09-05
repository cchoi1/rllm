#!/usr/bin/env python3
"""
Utility script to process chunked DeepCoder parquet files for Verl training.
This script helps avoid OOM errors by processing smaller chunks instead of one large file.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd


def list_available_chunks(chunks_dir: str) -> Dict[str, List[str]]:
    """List all available chunk files in the directory.
    
    Args:
        chunks_dir: Directory containing the chunked parquet files
        
    Returns:
        Dictionary with 'train' and 'test' keys, each containing list of chunk file paths
    """
    chunks_dir = Path(chunks_dir)
    if not chunks_dir.exists():
        print(f"Chunks directory not found: {chunks_dir}")
        return {}
    
    chunks = {"train": [], "test": []}
    
    for file_path in chunks_dir.glob("*.parquet"):
        filename = file_path.name
        if filename.startswith("train_chunk_"):
            chunks["train"].append(str(file_path))
        elif filename.startswith("test_chunk_"):
            chunks["test"].append(str(file_path))
    
    # Sort chunks by chunk number
    for split in chunks:
        chunks[split].sort(key=lambda x: int(x.split("chunk_")[1].split("_")[0]))
    
    return chunks


def list_verl_chunks(chunks_dir: str) -> Dict[str, List[str]]:
    """List all available Verl chunk files in the directory.
    
    Args:
        chunks_dir: Directory containing the chunked Verl parquet files
        
    Returns:
        Dictionary with 'train' and 'test' keys, each containing list of chunk file paths
    """
    chunks_dir = Path(chunks_dir)
    if not chunks_dir.exists():
        print(f"Chunks directory not found: {chunks_dir}")
        return {}
    
    chunks = {"train": [], "test": []}
    
    for file_path in chunks_dir.glob("*_verl_chunk_*.parquet"):
        filename = file_path.name
        if "train" in filename:
            chunks["train"].append(str(file_path))
        elif "test" in filename:
            chunks["test"].append(str(file_path))
    
    # Sort chunks by chunk number
    for split in chunks:
        chunks[split].sort(key=lambda x: int(x.split("chunk_")[1].split("_")[0]))
    
    return chunks


def convert_chunk_to_verl_format(chunk_path: str) -> List[Dict[str, Any]]:
    """Convert a chunk parquet file to Verl format.
    
    Args:
        chunk_path: Path to the chunk parquet file
        
    Returns:
        List of dictionaries in Verl format
    """
    print(f"Converting {chunk_path} to Verl format...")
    
    # Load the chunk
    df = pd.read_parquet(chunk_path)
    data = df.to_dict("records")
    
    # Convert to Verl format
    verl_data = []
    for entry in data:
        verl_entry = {
            "prompt": [{"role": "user", "content": entry["question"]}],
            "reward_model": {
                "style": "rule",
                "ground_truth": entry["ground_truth"],
            },
            "extra_info": entry,
        }
        verl_data.append(verl_entry)
    
    print(f"Converted {len(verl_data)} examples to Verl format")
    return verl_data


def save_verl_chunk(verl_data: List[Dict[str, Any]], output_path: str):
    """Save Verl-formatted data to a parquet file.
    
    Args:
        verl_data: List of Verl-formatted dictionaries
        output_path: Path to save the Verl-formatted parquet file
    """
    df = pd.DataFrame(verl_data)
    df.to_parquet(output_path)
    print(f"Saved Verl-formatted data to {output_path}")


def process_chunk_for_verl(chunk_path: str, output_dir: str = None) -> str:
    """Process a single chunk for Verl training.
    
    Args:
        chunk_path: Path to the input chunk parquet file
        output_dir: Directory to save the Verl-formatted file (None for same directory)
        
    Returns:
        Path to the generated Verl-formatted file
    """
    chunk_path = Path(chunk_path)
    
    if output_dir is None:
        output_dir = chunk_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
    
    # Convert to Verl format
    verl_data = convert_chunk_to_verl_format(str(chunk_path))
    
    # Generate output filename
    output_filename = chunk_path.stem + "_verl.parquet"
    output_path = output_dir / output_filename
    
    # Save Verl-formatted data
    save_verl_chunk(verl_data, str(output_path))
    
    return str(output_path)


def process_all_chunks_for_verl(chunks_dir: str, output_dir: str = None, split: str = None):
    """Process all chunks in a directory for Verl training.
    
    Args:
        chunks_dir: Directory containing chunked parquet files
        output_dir: Directory to save Verl-formatted files (None for same directory)
        split: Process only 'train' or 'test' chunks (None for both)
    """
    chunks = list_available_chunks(chunks_dir)
    
    if not chunks:
        print("No chunks found!")
        return
    
    print(f"Found chunks:")
    for split_name, chunk_files in chunks.items():
        print(f"  {split_name}: {len(chunk_files)} chunks")
    
    # Process chunks
    splits_to_process = [split] if split else ["train", "test"]
    
    for split_name in splits_to_process:
        if split_name not in chunks:
            print(f"No {split_name} chunks found, skipping...")
            continue
            
        print(f"\nProcessing {split_name} chunks...")
        for chunk_path in chunks[split_name]:
            process_chunk_for_verl(chunk_path, output_dir)


def validate_verl_chunk(chunk_path: str) -> bool:
    """Validate that a chunk file is in proper Verl format.
    
    Args:
        chunk_path: Path to the chunk file to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        df = pd.read_parquet(chunk_path)
        data = df.to_dict("records")
        
        if not data:
            print(f"Warning: {chunk_path} is empty")
            return False
        
        # Check if it's already in Verl format
        first_entry = data[0]
        if "prompt" in first_entry and "reward_model" in first_entry:
            print(f"✓ {chunk_path} is already in Verl format ({len(data)} examples)")
            return True
        elif "question" in first_entry and "ground_truth" in first_entry:
            print(f"⚠ {chunk_path} needs conversion to Verl format ({len(data)} examples)")
            return False
        else:
            print(f"✗ {chunk_path} has unknown format")
            return False
            
    except Exception as e:
        print(f"✗ Error reading {chunk_path}: {e}")
        return False


def validate_all_verl_chunks(chunks_dir: str):
    """Validate all chunk files in a directory.
    
    Args:
        chunks_dir: Directory containing chunk files
    """
    chunks_dir = Path(chunks_dir)
    if not chunks_dir.exists():
        print(f"Chunks directory not found: {chunks_dir}")
        return
    
    print(f"Validating chunks in {chunks_dir}...")
    
    valid_count = 0
    total_count = 0
    
    for file_path in chunks_dir.glob("*.parquet"):
        total_count += 1
        if validate_verl_chunk(str(file_path)):
            valid_count += 1
    
    print(f"\nValidation complete: {valid_count}/{total_count} files are valid")


def main():
    parser = argparse.ArgumentParser(description="Process chunked DeepCoder data for Verl training")
    parser.add_argument("--chunks-dir", type=str, required=True,
                       help="Directory containing chunked parquet files")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Directory to save Verl-formatted files (default: same as chunks)")
    parser.add_argument("--split", type=str, choices=["train", "test"], default=None,
                       help="Process only train or test chunks (default: both)")
    parser.add_argument("--list-only", action="store_true",
                       help="Only list available chunks without processing")
    parser.add_argument("--validate", action="store_true",
                       help="Validate chunk files instead of processing them")
    parser.add_argument("--list-verl", action="store_true",
                       help="List Verl-formatted chunks specifically")
    
    args = parser.parse_args()
    
    if args.list_only:
        if args.list_verl:
            chunks = list_verl_chunks(args.chunks_dir)
        else:
            chunks = list_available_chunks(args.chunks_dir)
            
        if chunks:
            print(f"\nAvailable chunks in {args.chunks_dir}:")
            for split_name, chunk_files in chunks.items():
                print(f"\n{split_name.upper()} chunks ({len(chunk_files)} files):")
                for chunk_path in chunk_files:
                    print(f"  {Path(chunk_path).name}")
        else:
            print("No chunks found!")
    elif args.validate:
        validate_all_verl_chunks(args.chunks_dir)
    else:
        process_all_chunks_for_verl(args.chunks_dir, args.output_dir, args.split)


if __name__ == "__main__":
    main()
