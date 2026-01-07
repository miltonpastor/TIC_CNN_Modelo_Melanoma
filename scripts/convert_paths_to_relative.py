#!/usr/bin/env python3
"""
Script to convert absolute paths in train/validation/test files to relative paths.
Converts paths like: /mnt/c/Users/.../ISIC/2020/train/ISIC_5432603.jpg
To relative paths like: ../ISIC/2020/train/ISIC_5432603.jpg
"""

import os
import argparse


def convert_paths(input_file, output_file):
    """
    Convert paths from absolute to relative starting from ISIC.
    
    Args:
        input_file: Path to input file with absolute paths
        output_file: Path to output file for relative paths
    """
    converted_count = 0
    
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            
            # Split path and label
            parts = line.rsplit(' ', 1)
            if len(parts) != 2:
                continue
            
            path, label = parts
            
            # Find the position of 'ISIC' in the path
            isic_pos = path.find('ISIC')
            if isic_pos != -1:
                # Extract from ISIC onwards
                new_path = path[isic_pos:]
                # Add the relative prefix
                new_path = f"data/{new_path}"
                f_out.write(f"{new_path} {label}\n")
                converted_count += 1
    
    return converted_count


def main():
    parser = argparse.ArgumentParser(
        description='Convert absolute paths to relative paths in dataset files'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='../data/lists',
        help='Directory containing input files (default: ../data/lists)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory for output files (default: same as input-dir)'
    )
    parser.add_argument(
        '--suffix',
        type=str,
        default='_relative',
        help='Suffix to add to output filenames (default: _relative)'
    )
    parser.add_argument(
        '--files',
        nargs='+',
        default=['train.txt', 'validation.txt', 'test.txt'],
        help='Files to convert (default: train.txt validation.txt test.txt)'
    )
    
    args = parser.parse_args()
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else args.input_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("Converting paths to relative format...")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {output_dir}")
    print("-" * 60)
    
    total_converted = 0
    
    # Convert all specified files
    for filename in args.files:
        input_file = os.path.join(args.input_dir, filename)
        
        if not os.path.exists(input_file):
            print(f"⚠️  Skipping {filename} (file not found)")
            continue
        
        # Generate output filename
        base_name = filename.replace('.txt', '')
        output_filename = f"{base_name}{args.suffix}.txt"
        output_file = os.path.join(output_dir, output_filename)
        
        # Convert paths
        count = convert_paths(input_file, output_file)
        total_converted += count
        
        print(f"✓ {filename} -> {output_filename} ({count:,} lines)")
    
    print("-" * 60)
    print(f"Total lines converted: {total_converted:,}")
    print("Done!")


if __name__ == '__main__':
    main()
