#!/usr/bin/env python3
"""
Checkpoint Cleanup Utility

This script helps clean up existing checkpoint folders to free up disk space.
It provides options to:
1. Remove all but the N most recent checkpoints
2. Remove checkpoints older than X days
3. Show checkpoint folder size and file information
4. Interactive cleanup with size estimates

Usage:
    python cleanup_checkpoints.py --help
    python cleanup_checkpoints.py --show-info
    python cleanup_checkpoints.py --keep-latest 3
    python cleanup_checkpoints.py --remove-older-than 7
"""

import os
import argparse
import glob
import shutil
from datetime import datetime, timedelta
from pathlib import Path


def get_file_size_mb(filepath):
    """Get file size in MB."""
    try:
        return os.path.getsize(filepath) / (1024 * 1024)
    except OSError:
        return 0


def get_checkpoint_info(checkpoint_dir):
    """Get information about all checkpoint files."""
    checkpoint_files = []
    
    # Look for common checkpoint patterns
    patterns = [
        "*.weights.h5",
        "ckpt_*.h5", 
        "checkpoint_*.h5",
        "epoch_*.h5"
    ]
    
    for pattern in patterns:
        files = glob.glob(os.path.join(checkpoint_dir, pattern))
        for filepath in files:
            stat = os.stat(filepath)
            checkpoint_files.append({
                'path': filepath,
                'name': os.path.basename(filepath),
                'size_mb': get_file_size_mb(filepath),
                'modified': datetime.fromtimestamp(stat.st_mtime),
                'age_days': (datetime.now() - datetime.fromtimestamp(stat.st_mtime)).days
            })
    
    # Sort by modification time (newest first)
    checkpoint_files.sort(key=lambda x: x['modified'], reverse=True)
    return checkpoint_files


def show_checkpoint_info(checkpoint_dir):
    """Display information about checkpoints."""
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory '{checkpoint_dir}' does not exist.")
        return
    
    checkpoint_files = get_checkpoint_info(checkpoint_dir)
    
    if not checkpoint_files:
        print(f"No checkpoint files found in '{checkpoint_dir}'")
        return
    
    total_size = sum(f['size_mb'] for f in checkpoint_files)
    
    print(f"\nCheckpoint Directory: {checkpoint_dir}")
    print(f"Total Files: {len(checkpoint_files)}")
    print(f"Total Size: {total_size:.1f} MB")
    print(f"Disk Usage: {total_size/1024:.2f} GB" if total_size > 1024 else f"Disk Usage: {total_size:.1f} MB")
    print("\nCheckpoint Files (newest first):")
    print("-" * 80)
    print(f"{'File Name':<40} {'Size (MB)':<10} {'Age (days)':<12} {'Modified':<20}")
    print("-" * 80)
    
    for f in checkpoint_files:
        print(f"{f['name']:<40} {f['size_mb']:<10.1f} {f['age_days']:<12} {f['modified'].strftime('%Y-%m-%d %H:%M'):<20}")


def cleanup_keep_latest(checkpoint_dir, keep_count, dry_run=True):
    """Remove all but the N most recent checkpoints."""
    checkpoint_files = get_checkpoint_info(checkpoint_dir)
    
    if len(checkpoint_files) <= keep_count:
        print(f"Only {len(checkpoint_files)} checkpoints found. No cleanup needed (keeping {keep_count}).")
        return
    
    files_to_remove = checkpoint_files[keep_count:]
    total_freed_mb = sum(f['size_mb'] for f in files_to_remove)
    
    print(f"\nWould remove {len(files_to_remove)} checkpoint files to keep {keep_count} most recent:")
    print(f"Total space to be freed: {total_freed_mb:.1f} MB")
    print("\nFiles to be removed:")
    for f in files_to_remove:
        print(f"  - {f['name']} ({f['size_mb']:.1f} MB, {f['age_days']} days old)")
    
    if dry_run:
        print(f"\n[DRY RUN] Use --execute to actually remove files.")
        return
    
    # Confirm before deletion
    response = input(f"\nAre you sure you want to delete {len(files_to_remove)} files? (yes/no): ")
    if response.lower() != 'yes':
        print("Operation cancelled.")
        return
    
    # Remove files
    removed_count = 0
    actual_freed_mb = 0
    for f in files_to_remove:
        try:
            os.remove(f['path'])
            removed_count += 1
            actual_freed_mb += f['size_mb']
            print(f"Removed: {f['name']}")
        except OSError as e:
            print(f"Error removing {f['name']}: {e}")
    
    print(f"\nCleanup complete!")
    print(f"Removed {removed_count} files")
    print(f"Freed {actual_freed_mb:.1f} MB of disk space")


def cleanup_older_than(checkpoint_dir, days, dry_run=True):
    """Remove checkpoints older than N days."""
    checkpoint_files = get_checkpoint_info(checkpoint_dir)
    cutoff_date = datetime.now() - timedelta(days=days)
    
    files_to_remove = [f for f in checkpoint_files if f['modified'] < cutoff_date]
    
    if not files_to_remove:
        print(f"No checkpoint files older than {days} days found.")
        return
    
    total_freed_mb = sum(f['size_mb'] for f in files_to_remove)
    
    print(f"\nWould remove {len(files_to_remove)} checkpoint files older than {days} days:")
    print(f"Total space to be freed: {total_freed_mb:.1f} MB")
    print(f"Cutoff date: {cutoff_date.strftime('%Y-%m-%d %H:%M')}")
    print("\nFiles to be removed:")
    for f in files_to_remove:
        print(f"  - {f['name']} ({f['size_mb']:.1f} MB, {f['age_days']} days old)")
    
    if dry_run:
        print(f"\n[DRY RUN] Use --execute to actually remove files.")
        return
    
    # Confirm before deletion
    response = input(f"\nAre you sure you want to delete {len(files_to_remove)} files? (yes/no): ")
    if response.lower() != 'yes':
        print("Operation cancelled.")
        return
    
    # Remove files
    removed_count = 0
    actual_freed_mb = 0
    for f in files_to_remove:
        try:
            os.remove(f['path'])
            removed_count += 1
            actual_freed_mb += f['size_mb']
            print(f"Removed: {f['name']}")
        except OSError as e:
            print(f"Error removing {f['name']}: {e}")
    
    print(f"\nCleanup complete!")
    print(f"Removed {removed_count} files")
    print(f"Freed {actual_freed_mb:.1f} MB of disk space")


def main():
    parser = argparse.ArgumentParser(
        description="Checkpoint cleanup utility for managing disk space",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show checkpoint information
  python cleanup_checkpoints.py --show-info
  
  # Keep only the 3 most recent checkpoints (dry run)
  python cleanup_checkpoints.py --keep-latest 3
  
  # Actually remove files (keep 3 most recent)  
  python cleanup_checkpoints.py --keep-latest 3 --execute
  
  # Remove checkpoints older than 7 days
  python cleanup_checkpoints.py --remove-older-than 7 --execute
        """
    )
    
    parser.add_argument(
        "--checkpoint-dir", 
        default="./checkpoints",
        help="Path to checkpoint directory (default: ./checkpoints)"
    )
    
    parser.add_argument(
        "--show-info",
        action="store_true", 
        help="Show information about checkpoint files"
    )
    
    parser.add_argument(
        "--keep-latest",
        type=int,
        help="Keep only the N most recent checkpoints"
    )
    
    parser.add_argument(
        "--remove-older-than",
        type=int,
        help="Remove checkpoints older than N days"
    )
    
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform the cleanup (default is dry-run)"
    )
    
    args = parser.parse_args()
    
    # Show help if no action specified
    if not any([args.show_info, args.keep_latest, args.remove_older_than]):
        parser.print_help()
        return
    
    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    
    if args.show_info:
        show_checkpoint_info(checkpoint_dir)
    
    if args.keep_latest:
        cleanup_keep_latest(checkpoint_dir, args.keep_latest, dry_run=not args.execute)
    
    if args.remove_older_than:
        cleanup_older_than(checkpoint_dir, args.remove_older_than, dry_run=not args.execute)


if __name__ == "__main__":
    main()