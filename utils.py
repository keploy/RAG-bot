import os
import hashlib
import pickle
from pathlib import Path
from typing import Tuple


def compute_file_checksum(file_path):
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def load_checksums(checksums_file):
    if Path(checksums_file).exists():
        with open(checksums_file, "rb") as f:
            return pickle.load(f)
    return None


def save_checksums(checksums, checksums_file):
    with open(checksums_file, "wb") as f:
        pickle.dump(checksums, f)


# Function to get all MDX files from the docs directory and its subdirectories
def get_mdx_files(directory):
    print(f"Searching for MDX files in {directory}...")
    mdx_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".md"):
                mdx_files.append(os.path.join(root, file))
    print(f"Found {len(mdx_files)} MDX files.")
    return mdx_files


# Detect changes in documentation files.
def detect_docs_changes(mdx_file_paths, last_checksums) -> Tuple[bool, dict[str, str]]:
    current_checksums = {
        file_path: compute_file_checksum(file_path) for file_path in mdx_file_paths
    }

    # when the checksums file doesn't exist
    if last_checksums is None:
        return True, current_checksums

    # if a file is added/removed
    if set(current_checksums.keys()) != set(last_checksums.keys()):
        return True, current_checksums

    # if a file has been modified
    for file_path in current_checksums:
        if current_checksums[file_path] != last_checksums.get(file_path):
            return True, current_checksums

    # No changes
    return False, current_checksums
