# burrhole_detection_selfannotated

expects:
data/train/preop.nii.gz
data/train/postop.nii.gz

python register.py data/train
-> data/train/poop_to_preop.tfm

python subtract.py data/train
-> data/train/diff.nii.gz

python binarize.py data/train
-> data/train/burrhole_mask_autoannot.nii.gz

python export_for_nnunet.py data nnunet_dataset_path