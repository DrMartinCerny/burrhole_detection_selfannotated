#!/usr/bin/env python3
import sys
import os
import json
from pathlib import Path
import shutil
from tqdm import tqdm


def find_cases(root: Path):
    """
    Recursively find all subfolders under `root` that contain:
      - preop.nii.gz
      - burrhole_mask_autoannot.nii.gz
    """
    cases = []
    for dirpath, dirnames, filenames in os.walk(root):
        fset = set(filenames)
        if "preop.nii.gz" in fset and "burrhole_mask_autoannot.nii.gz" in fset:
            cases.append(Path(dirpath))
    return sorted(cases)


def ensure_dir(p: Path):
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)


def create_dataset_json(target_root: Path, num_cases: int):
    """
    Create a minimal dataset.json for nnU-Net v2 compatibility.
    """
    dataset_json = {
        "labels": {
            "0": "background",
            "1": "burrhole"
        },
        "numTraining": num_cases,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
        "modality": {
            "0": "CT"
        }
    }

    json_path = target_root / "dataset.json"
    with open(json_path, "w") as f:
        json.dump(dataset_json, f, indent=4)

    print(f"[INFO] Wrote dataset.json with {num_cases} training cases.")


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <source_root> <target_root>")
        sys.exit(1)

    source_root = Path(sys.argv[1]).expanduser().resolve()
    target_root = Path(sys.argv[2]).expanduser().resolve()

    if not source_root.is_dir():
        print(f"Error: {source_root} is not a directory.")
        sys.exit(1)

    ensure_dir(target_root)

    imagesTr = target_root / "imagesTr"
    labelsTr = target_root / "labelsTr"

    ensure_dir(imagesTr)
    ensure_dir(labelsTr)

    # -----------------------
    # Find valid cases
    # -----------------------
    case_dirs = find_cases(source_root)
    print(f"Found {len(case_dirs)} valid cases.")

    # -----------------------
    # Export each case
    # -----------------------
    for idx, case_dir in enumerate(tqdm(case_dirs, desc="Exporting to nnUNet")):
        case_id = f"case_{idx:04d}"

        preop_src = case_dir / "preop.nii.gz"
        label_src = case_dir / "burrhole_mask_autoannot.nii.gz"

        # nnUNet naming:
        # CT image → caseID_0000.nii.gz   (0000 = modality index)
        # label    → caseID.nii.gz
        img_dst = imagesTr / f"{case_id}_0000.nii.gz"
        label_dst = labelsTr / f"{case_id}.nii.gz"

        shutil.copy(preop_src, img_dst)
        shutil.copy(label_src, label_dst)

    # -----------------------
    # Write dataset.json
    # -----------------------
    create_dataset_json(target_root, len(case_dirs))

    print("[SUCCESS] Export finished.")


if __name__ == "__main__":
    main()
