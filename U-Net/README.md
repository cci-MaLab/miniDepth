# U-Net (legacy folder)

This folder contains the original U-Net training and inference scripts before the project was reorganized.

## Files

- config.py: Hyperparameters and loss configuration.
- data_loader.py: Patch sampling dataset and augmentation utilities.
- Pre_Process_Project.py: TIFF preprocessing pipeline.
- Unet.py: U-Net architecture, losses, metrics, and logging helpers.
- train.py: Training entry script.
- predict.py: Inference and threshold/evaluation helpers.
- Region_Growing_Method.py: Classical region-growing segmentation alternative.

## Run (legacy scripts)

From the TOOL root:

python U-Net/train.py
python U-Net/predict.py

## Important note

The actively maintained package layout is:

- segmentation/: current training/inference code
- global_mask_pipeline/: cross-plane global mask generation

If you are starting new work, prefer using the newer module commands:

python -m segmentation.train
python -m segmentation.predict
python -m global_mask_pipeline.save_global_plane_masks
