"""
Utility functions for processing the raw food dataset.

This script contains the logic to scan directories for images and JSON files,
extract image-text pairs, normalize text, and build a consolidated
`docstore.parquet` file and a training CSV file from the raw data.
"""
import os
import re
import json
import hashlib
from glob import glob
from typing import List, Dict, Any, Iterable, Tuple

import pandas as pd
from PIL import Image


def _normalize_title(s: str) -> str:
    """Normalizes a string by lowercasing, removing special characters, and standardizing Persian characters."""
    if s is None: return ""
    s = str(s).strip().replace("ي", "ی").replace("ك", "ک")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\u0600-\u06FF\s-]", "", s)
    return s.lower()


def _iter_json_records(json_path: str) -> Iterable[Dict[str, Any]]:
    """Iterates through JSON records in a file, handling both single/multi-object formats."""
    with open(json_path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
    if not txt: return
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict): yield obj; return
        for it in obj if isinstance(obj, list) else []:
            if isinstance(it, dict): yield it
        return
    except json.JSONDecodeError:
        pass
    for line in txt.splitlines():
        if not (line := line.strip()): continue
        try:
            if isinstance((obj := json.loads(line)), dict): yield obj
        except json.JSONDecodeError:
            continue


def _collect_pairs(root: str) -> pd.DataFrame:
    """Scans a root directory to find and collect all valid image-text pairs from JSON files."""
    rows: List[Dict[str, Any]] = []
    json_files = glob(os.path.join(root, "**/*.json"), recursive=True)
    if not json_files:
        raise RuntimeError(f"No JSON files found recursively in {root}")
    for jp in json_files:
        base_dir = os.path.dirname(jp)
        for rec in _iter_json_records(jp):
            title, resp, img_rel = rec.get("title"), rec.get("response"), rec.get("image_path")
            if not all([title, resp, img_rel]): continue
            img_abs = os.path.normpath(os.path.join(base_dir, img_rel))
            if not os.path.isfile(img_abs):
                print(f"[Warn] Missing image file, skipping: {img_abs}")
                continue

            canon_title = _normalize_title(title)
            folder_tag = os.path.basename(os.path.dirname(img_abs))
            stub = f"{folder_tag}\u241F{canon_title}\u241F{os.path.basename(img_abs)}"
            group_id = hashlib.sha1(stub.encode("utf-8")).hexdigest()[:16]
            dedupe_key = f"{canon_title}@@{folder_tag}"

            rows.append({
                "title": str(title),
                "text": str(resp),
                "image_path": img_abs,
                "canonical_dish": group_id,
                "dedupe_key": dedupe_key,
            })
    return pd.DataFrame(rows)


def _build_docstore(df: pd.DataFrame) -> pd.DataFrame:
    """Builds the final docstore, handling deduplication and ID generation."""
    # Deduplicate based on the passage text to ensure each entry is unique.
    df = df.drop_duplicates(subset=['text'], keep='first').reset_index(drop=True)

    def _mk_id(row):
        base = row["canonical_dish"] + "\u241F" + row["text"][:32]
        return hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]

    return pd.DataFrame({
        "id": df.apply(_mk_id, axis=1),
        "food_id": df["dedupe_key"],
        "passage_text": df["text"],
        "title": df["title"],
    })


def prep_dataset(root: str, out_dir: str) -> Tuple[str, str]:
    """
    Main function to prepare the dataset.

    It scans the raw data directory, collects image-text pairs, creates a
    deduplicated docstore, and saves both a training CSV and the docstore
    to the specified output directory.

    Args:
        root (str): The root directory of the raw dataset.
        out_dir (str): The directory to save the processed files.

    Returns:
        Tuple[str, str]: A tuple containing the path to the training CSV
                         and the path to the docstore parquet file.
    """
    print("Starting dataset preparation...")
    os.makedirs(out_dir, exist_ok=True)

    df_pairs = _collect_pairs(root)

    # Filter out unreadable images before saving the training CSV
    mask = df_pairs["image_path"].apply(lambda p: os.path.isfile(p))
    df_pairs = df_pairs[mask].reset_index(drop=True)

    docstore = _build_docstore(df_pairs)

    # Save the files
    train_csv_path = os.path.join(out_dir, "train_pairs.csv")
    docstore_path = os.path.join(out_dir, "docstore.parquet")

    df_pairs[["image_path", "text", "canonical_dish", "dedupe_key"]].to_csv(train_csv_path, index=False)
    docstore.to_parquet(docstore_path, index=False)

    print(f"Dataset preparation complete. {len(df_pairs)} training pairs saved.")
    print(f"Docstore with {len(docstore)} unique passages created.")

    return train_csv_path, docstore_path
