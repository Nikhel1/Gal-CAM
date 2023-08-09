# Gal-CAM
The official code of PASA 2023 paper (Deep Learning for Morphological Identification of Extended Radio Galaxies using Weak Labels). 

## Prerequisite
- Python 3.6, PyTorch 1.9, and others in environment.yml
- You can create the environment from environment.yml file
```
conda env create -f environment.yml
```

## Usage
### Step 1. Prepare dataset.
- Download link for the data will be available soon, the data is in the MS COCO format.
- Generate mask from annotations (annToMask.py file in ./mscoco/). The annotation files will be available soon.
- Download MS COCO image-level labels are present in ./mscoco/
### Step 2. Train ReCAM and generate seeds.
- Please specify a workspace to save the model and logs.
```
CUDA_VISIBLE_DEVICES=0 python run_sample_gal.py --mscoco_root ../dataset/ --work_space YOUR_WORK_SPACE --train_cam_pass True --train_recam_pass True --make_recam_pass True --eval_cam_pass True
```
### Step 3. Train IRN and generate instance segmentation masks.
```
CUDA_VISIBLE_DEVICES=0 python run_sample_gal.py --mscoco_root ../dataset/ --work_space YOUR_WORK_SPACE --cam_to_ir_label_pass True --train_irn_pass True --make_ins_seg_pass True
```

## Acknowledgment
Different parts of the code are borrowed from [IRN](https://github.com/jiwoon-ahn/irn), [AdvCAM](https://github.com/jbeomlee93/AdvCAM) and [ReCAM](https://github.com/zhaozhengChen/ReCAM).
