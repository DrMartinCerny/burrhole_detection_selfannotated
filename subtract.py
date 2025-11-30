#!/usr/bin/env python3
import sys
import os
from pathlib import Path

import SimpleITK as sitk
from tqdm import tqdm


def find_case_dirs(root: Path):
    """
    Recursively find all subfolders under `root` that contain
    preop.nii.gz, postop.nii.gz and postop_to_preop.tfm.
    """
    case_dirs = []
    for dirpath, dirnames, filenames in os.walk(root):
        filenames = set(filenames)
        if (
            "preop.nii.gz" in filenames and
            "postop.nii.gz" in filenames and
            "postop_to_preop.tfm" in filenames
        ):
            case_dirs.append(Path(dirpath))
    return sorted(case_dirs)


def subtract_postop_from_preop(preop_path: Path,
                               postop_path: Path,
                               transform_path: Path,
                               output_diff_path: Path):
    """
    Load preop (fixed), postop (moving), apply existing transform to postop,
    subtract postop from preop, save as diff.nii.gz.

    preop, postop, and the transformed postop are treated as 3D images.
    """
    # Read fixed image as float
    preop = sitk.ReadImage(str(preop_path), sitk.sitkFloat32)

    # Read moving image as float
    postop = sitk.ReadImage(str(postop_path), sitk.sitkFloat32)

    # Read transform
    transform = sitk.ReadTransform(str(transform_path))

    # Resample postop into preop space (overwrite postop variable, no need to keep original)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(preop)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)

    postop = resampler.Execute(postop)  # postop is now transformed version

    # Subtract postop from preop (preop - postop)
    diff = preop - postop

    # Save difference image
    sitk.WriteImage(diff, str(output_diff_path))


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <root_folder>")
        sys.exit(1)

    root = Path(sys.argv[1]).expanduser().resolve()
    if not root.is_dir():
        print(f"Error: {root} is not a directory.")
        sys.exit(1)

    case_dirs = find_case_dirs(root)
    if not case_dirs:
        print("No folders with preop.nii.gz, postop.nii.gz and postop_to_preop.tfm found.")
        sys.exit(0)

    print(f"Found {len(case_dirs)} case folders.")

    for case_dir in tqdm(case_dirs, desc="Computing differences"):
        preop_path = case_dir / "preop.nii.gz"
        postop_path = case_dir / "postop.nii.gz"
        transform_path = case_dir / "postop_to_preop.tfm"
        output_diff_path = case_dir / "diff.nii.gz"

        # Skip if already exists
        if output_diff_path.exists():
            continue

        try:
            subtract_postop_from_preop(
                preop_path,
                postop_path,
                transform_path,
                output_diff_path
            )
        except Exception as e:
            print(f"\n[WARN] Failed for {case_dir}: {e}")


if __name__ == "__main__":
    main()
