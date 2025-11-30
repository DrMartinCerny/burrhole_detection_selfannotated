#!/usr/bin/env python3
import sys
import os
from pathlib import Path

import SimpleITK as sitk
from tqdm import tqdm


# ==========================
# CONFIG (tweak if needed)
# ==========================

BONE_THRESHOLD_HU = 300.0      # voxels above this in preop are treated as bone
MIN_COMPONENT_SIZE = 50        # min number of voxels for connected components

HEAD_CAP_DEPTH_MM = 100.0      # keep only candidates within top X mm from vertex


def find_case_dirs(root: Path):
    """
    Recursively find all subfolders under `root` that contain
    preop.nii.gz and diff.nii.gz.
    """
    case_dirs = []
    for dirpath, dirnames, filenames in os.walk(root):
        filenames = set(filenames)
        if "preop.nii.gz" in filenames and "diff.nii.gz" in filenames:
            case_dirs.append(Path(dirpath))
    return sorted(case_dirs)


def compute_head_cap_mask(reference_image: sitk.Image) -> sitk.Image:
    """
    Create a binary mask (0/1) that is 1 only in the 'head cap':
    the top HEAD_CAP_DEPTH_MM from the vertex, along the
    superior–inferior axis, using image spacing + direction.
    """
    size = reference_image.GetSize()        # (nx, ny, nz)
    spacing = reference_image.GetSpacing()  # in mm
    direction = reference_image.GetDirection()

    # Direction is a flat list of 9 values -> 3x3 matrix
    # Each axis j has direction vector v_j = (d[3*j], d[3*j+1], d[3*j+2])
    v0 = direction[0:3]
    v1 = direction[3:6]
    v2 = direction[6:9]
    axes = [v0, v1, v2]

    # Find which index axis corresponds most to the superior-inferior (z) axis
    # In LPS, z is the "S" direction (third component).
    si_axis = max(range(3), key=lambda j: abs(axes[j][2]))
    si_dir = axes[si_axis][2]  # sign: +1 ~ index increases toward superior, -1 toward inferior

    # Determine top slice index (vertex side) and number of slices to keep
    n_slices = size[si_axis]
    depth_slices = int(HEAD_CAP_DEPTH_MM / spacing[si_axis])

    if depth_slices < 1:
        depth_slices = 1
    if depth_slices > n_slices:
        depth_slices = n_slices

    if si_dir > 0:
        # index increases toward superior: top = last slice
        start_slice = max(0, n_slices - depth_slices)
        end_slice = n_slices - 1
    else:
        # index increases toward inferior: top = first slice
        start_slice = 0
        end_slice = min(depth_slices - 1, n_slices - 1)

    # Create a mask that is 1 only in [start_slice, end_slice] along si_axis
    mask = sitk.Image(size, sitk.sitkUInt8)
    mask.CopyInformation(reference_image)

    # We’ll fill it with 1s slice-wise
    arr = sitk.GetArrayFromImage(mask)  # note: array is z, y, x

    # Map si_axis (0:x,1:y,2:z) to numpy axis in z,y,x order
    # SimpleITK GetArrayFromImage returns [k,z-axis][j,y][i,x]:
    # so numpy_axis_for_sitk = {0: 2, 1:1, 2:0}
    axis_map = {0: 2, 1: 1, 2: 0}
    np_axis = axis_map[si_axis]

    # Build a slice object to select the head cap range along np_axis
    slicer = [slice(None), slice(None), slice(None)]
    slicer[np_axis] = slice(start_slice, end_slice + 1)
    arr[tuple(slicer)] = 1

    mask = sitk.GetImageFromArray(arr)
    mask.CopyInformation(reference_image)
    return mask


def create_burrhole_mask(preop_path: Path,
                         diff_path: Path,
                         output_mask_path: Path):
    """
    Create a binary mask of burr-hole localization from:
      - preop CT (preop_path)
      - diff image (preop - transformed_postop, diff_path)

    Pipeline:
      1) Threshold preop to get bone mask.
      2) Keep only positive differences (bone lost).
      3) Restrict diff to bone region.
      4) Otsu threshold to get candidate regions.
      5) Morphological cleanup + min component size.
      6) Restrict to top HEAD_CAP_DEPTH_MM of the head (head cap).
      7) Save as burrhole_mask_auto.nii.gz.
    """
    # Load images as float
    preop = sitk.ReadImage(str(preop_path), sitk.sitkFloat32)
    diff = sitk.ReadImage(str(diff_path), sitk.sitkFloat32)

    # 1) Bone mask from preop CT (HU threshold)
    bone_mask = sitk.BinaryThreshold(
        preop,
        lowerThreshold=BONE_THRESHOLD_HU,
        upperThreshold=1e9,
        insideValue=1,
        outsideValue=0,
    )

    # 2) Keep only positive diff (preop > postop)
    diff_pos = sitk.Clamp(diff, lowerBound=0.0)

    # 3) Restrict to bone region
    diff_bone = diff_pos * sitk.Cast(bone_mask, sitk.sitkFloat32)

    # 4) Otsu threshold on diff restricted to bone
    candidate_mask = sitk.OtsuThreshold(diff_bone, 0, 1)  # 0/1

    # 5a) Morphological closing to fill small gaps
    candidate_mask = sitk.BinaryMorphologicalClosing(
        candidate_mask,
        [1, 1, 1]
    )

    # 5b) Remove tiny connected components
    cc = sitk.ConnectedComponent(candidate_mask)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)

    cleaned_mask = sitk.Image(candidate_mask.GetSize(), sitk.sitkUInt8)
    cleaned_mask.CopyInformation(candidate_mask)

    for label in stats.GetLabels():
        size = stats.GetNumberOfPixels(label)
        if size >= MIN_COMPONENT_SIZE:
            cleaned_mask = cleaned_mask | sitk.Cast(cc == label, sitk.sitkUInt8)

    cleaned_mask = sitk.BinaryThreshold(cleaned_mask, 1, 255, 1, 0)

    # 6) Restrict to head cap (top HEAD_CAP_DEPTH_MM)
    head_cap_mask = compute_head_cap_mask(preop)
    final_mask = cleaned_mask * head_cap_mask  # logical AND in 0/1

    # 7) Save final burr-hole mask
    sitk.WriteImage(final_mask, str(output_mask_path))


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
        print("No folders with preop.nii.gz and diff.nii.gz found.")
        sys.exit(0)

    print(f"Found {len(case_dirs)} case folders.")

    for case_dir in tqdm(case_dirs, desc="Binarizing burr-hole masks"):
        preop_path = case_dir / "preop.nii.gz"
        diff_path = case_dir / "diff.nii.gz"
        output_mask_path = case_dir / "burrhole_mask_autoannot.nii.gz"

        if output_mask_path.exists():
            continue

        try:
            create_burrhole_mask(preop_path, diff_path, output_mask_path)
        except Exception as e:
            print(f"\n[WARN] Failed for {case_dir}: {e}")


if __name__ == "__main__":
    main()
