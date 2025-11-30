#!/usr/bin/env python3
import sys
import os
from pathlib import Path

import SimpleITK as sitk
from tqdm import tqdm


# ==========================
# CONFIG (tweak if needed)
# ==========================

# Bone HU thresholding for skull mask
BONE_LOWER_HU = 300.0      # start of bone
BONE_UPPER_HU = 3000.0     # sanity cap

# Morphology / cleanup of skull masks
MASK_CLOSING_RADIUS = 1    # in voxels for BinaryMorphologicalClosing
MIN_SKULL_COMPONENT_SIZE = 500  # min voxels to keep connected component

# Registration parameters
METRIC_SAMPLING_PERCENTAGE = 0.2
NUMBER_OF_ITERATIONS = 300


def find_case_dirs(root: Path):
    """
    Recursively find all subfolders under `root` that contain both
    preop.nii.gz and postop.nii.gz.
    """
    case_dirs = []
    for dirpath, dirnames, filenames in os.walk(root):
        filenames = set(filenames)
        if "preop.nii.gz" in filenames and "postop.nii.gz" in filenames:
            case_dirs.append(Path(dirpath))
    return sorted(case_dirs)


def create_skull_mask(image: sitk.Image) -> sitk.Image:
    """
    Create a skull mask from a CT image using HU thresholding and
    simple morphological cleanup.

    Returns a binary mask (0/1, sitkUInt8) roughly representing the skull.
    """
    # Threshold for bone
    mask = sitk.BinaryThreshold(
        image,
        lowerThreshold=BONE_LOWER_HU,
        upperThreshold=BONE_UPPER_HU,
        insideValue=1,
        outsideValue=0,
    )

    # Optional closing to fill small gaps in the skull
    if MASK_CLOSING_RADIUS > 0:
        mask = sitk.BinaryMorphologicalClosing(
            mask,
            [MASK_CLOSING_RADIUS] * image.GetDimension()
        )

    # Keep only reasonably large connected components (skull + possibly mandible)
    cc = sitk.ConnectedComponent(mask)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)

    cleaned = sitk.Image(mask.GetSize(), sitk.sitkUInt8)
    cleaned.CopyInformation(mask)

    for label in stats.GetLabels():
        size = stats.GetNumberOfPixels(label)
        if size >= MIN_SKULL_COMPONENT_SIZE:
            cleaned = cleaned | sitk.Cast(cc == label, sitk.sitkUInt8)

    # Final binary 0/1
    cleaned = sitk.BinaryThreshold(cleaned, 1, 255, 1, 0)
    return cleaned


def register_and_resample(preop_path: Path,
                          postop_path: Path,
                          output_image_path: Path,
                          output_transform_path: Path):
    """
    Register postop (moving) to preop (fixed) using SimpleITK with skull masks
    and save the resampled image + transform.
    Assumes 3D CT images.
    """
    # Read images as float32
    fixed = sitk.ReadImage(str(preop_path), sitk.sitkFloat32)
    moving = sitk.ReadImage(str(postop_path), sitk.sitkFloat32)

    if fixed.GetDimension() != 3 or moving.GetDimension() != 3:
        raise ValueError("Expected 3D images only.")

    # Create skull masks to drive registration on bone only
    fixed_mask = create_skull_mask(fixed)
    moving_mask = create_skull_mask(moving)

    # Initial transform: 3D Euler, centered on image geometry
    initial_transform = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    # Set up registration
    registration_method = sitk.ImageRegistrationMethod()

    # Metric: Mattes MI, masked to skull regions
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(METRIC_SAMPLING_PERCENTAGE)

    registration_method.SetMetricFixedMask(fixed_mask)
    registration_method.SetMetricMovingMask(moving_mask)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer: regular step gradient descent
    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-4,
        numberOfIterations=NUMBER_OF_ITERATIONS,
        gradientMagnitudeTolerance=1e-8,
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Multi-resolution pyramid
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Use rigid transform (Euler3D)
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # Run registration
    final_transform = registration_method.Execute(fixed, moving)

    # Optionally you can inspect final metric / number of iterations:
    # print("Final metric:", registration_method.GetMetricValue())
    # print("Optimizer iterations:", registration_method.GetOptimizerIteration())

    # Resample moving into fixed space
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetTransform(final_transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)

    moving_resampled = resampler.Execute(moving)

    # Save outputs
    sitk.WriteImage(moving_resampled, str(output_image_path))
    sitk.WriteTransform(final_transform, str(output_transform_path))


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
        print("No folders with preop.nii.gz and postop.nii.gz found.")
        sys.exit(0)

    print(f"Found {len(case_dirs)} case folders.")

    for case_dir in tqdm(case_dirs, desc="Registering cases (skull-masked rigid)"):
        preop_path = case_dir / "preop.nii.gz"
        postop_path = case_dir / "postop.nii.gz"
        output_image_path = case_dir / "postop_transformed.nii.gz"
        output_transform_path = case_dir / "postop_to_preop.tfm"

        # Skip if both outputs already exist
        if output_image_path.exists() and output_transform_path.exists():
            continue

        try:
            register_and_resample(
                preop_path,
                postop_path,
                output_image_path,
                output_transform_path
            )
        except Exception as e:
            print(f"\n[WARN] Failed for {case_dir}: {e}")


if __name__ == "__main__":
    main()
