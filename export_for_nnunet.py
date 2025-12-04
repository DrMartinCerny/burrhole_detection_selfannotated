#!/usr/bin/env python3
import sys
import os
import json
from pathlib import Path
import shutil
from tqdm import tqdm

def find_cases_with_label(root: Path):
    """
    Find all case folders under `root` that contain:
      - preop.nii.gz
      - burrhole_mask_autoannot.nii.gz
    Assumes structure: root/{CASE_ID}/...
    """
    cases = []
    if not root.is_dir():
        return cases

    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        files = {f.name for f in child.iterdir() if f.is_file()}
        if "preop.nii.gz" in files and "burrhole_mask_autoannot.nii.gz" in files:
            cases.append(child)
    return cases


def find_cases_without_label(root: Path):
    """
    Find all case folders under `root` that contain at least:
      - preop.nii.gz
    We ignore whether a label exists (nnU-Net test set is images-only).
    """
    cases = []
    if not root.is_dir():
        return cases

    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        files = {f.name for f in child.iterdir() if f.is_file()}
        if "preop.nii.gz" in files:
            cases.append(child)
    return cases

def ensure_dir(p: Path):
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)


def create_dataset_json(target_root: Path, num_training: int):
    """
    Create a minimal dataset.json for nnU-Net v2 compatibility.
    """
    dataset_json = {
        "labels": {
            "background": 0,
            "burrhole": 1
        },
        "channel_names": {
            "0": "CT"
        },
        "numTraining": num_training,
        "file_ending": ".nii.gz"
    }

    json_path = target_root / "dataset.json"
    with open(json_path, "w") as f:
        json.dump(dataset_json, f, indent=4)

    print(f"[INFO] Wrote dataset.json with {num_training} training cases.")


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <source_root> <target_root>")
        sys.exit(1)

    # source_root = {DATASET_PATH} which contains train/ and test/
    source_root = Path(sys.argv[1]).expanduser().resolve()
    target_root = Path(sys.argv[2]).expanduser().resolve()

    if not source_root.is_dir():
        print(f"Error: {source_root} is not a directory.")
        sys.exit(1)

    train_root = source_root / "train"
    test_root = source_root / "test"

    if not train_root.is_dir():
        print(f"Error: {train_root} does not exist or is not a directory.")
        sys.exit(1)

    ensure_dir(target_root)

    imagesTr = target_root / "imagesTr"
    labelsTr = target_root / "labelsTr"
    imagesTs = target_root / "imagesTs"

    ensure_dir(imagesTr)
    ensure_dir(labelsTr)
    ensure_dir(imagesTs)

    # -----------------------
    # Find training and test cases
    # -----------------------
    train_case_dirs = find_cases_with_label(train_root)
    test_case_dirs = find_cases_without_label(test_root)

    print(f"Found {len(train_case_dirs)} training cases with labels.")
    print(f"Found {len(test_case_dirs)} test cases (images only).")

    # -----------------------
    # Export training cases -> imagesTr + labelsTr
    # -----------------------
    for case_dir in tqdm(train_case_dirs, desc="Exporting TRAIN to nnUNet"):
        case_id = case_dir.name  # keep original numeric ID

        preop_src = case_dir / "preop.nii.gz"
        label_src = case_dir / "burrhole_mask_autoannot.nii.gz"

        img_dst = imagesTr   / f"{case_id}_0000.nii.gz"
        label_dst = labelsTr / f"{case_id}.nii.gz"

        shutil.copy(preop_src, img_dst)
        shutil.copy(label_src, label_dst)

    # -----------------------
    # Export test cases -> imagesTs (no labels)
    # -----------------------
    for case_dir in tqdm(test_case_dirs, desc="Exporting TEST to nnUNet"):
        case_id = case_dir.name  # keep original numeric ID

        preop_src = case_dir / "preop.nii.gz"
        img_dst = imagesTs / f"{case_id}_0000.nii.gz"

        shutil.copy(preop_src, img_dst)

    # -----------------------
    # Write dataset.json (numTraining = #train cases)
    # -----------------------
    create_dataset_json(target_root, len(train_case_dirs))

    print("[SUCCESS] Export finished.")


if __name__ == "__main__":
    main()
