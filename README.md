# DehazeFlow
Code and trained models for reproducing results of: 

**DehazeFlow: Multi-scale Conditional Flow Network for Single Image Dehazing<br>Hongyu Li, Jia Li, Dong Zhao, Long Xu<br>ACM Conference on Multimedia (ACM MM), 2021.
<br><br>**
**[Paper will come soon]**

## News
The latest performance of our method after parameter adjustment is as follows:

|        | PSNR   | SSIM   | LPIPS  |
| :----: | :----: | :----: |:----:  |
| indoor | 40.88  | 0.9897 | 0.0025 |
| outdoor| 34.50  | 0.9859 | 0.0051 |

Link of trained models: https://drive.google.com/drive/folders/1NtQyK5dVu47E2LBk5ETcivPO96YFomPN?usp=sharing

**Note: Opencv of this version is different from before. You need to run:**
```
conda uninstall opencv
pip install opencv-python==4.5.3.56
```

## Environment

* python==3.8.0
* lpips==0.1.3
* pytorch==1.8.0
* scikit-image==0.18.1
* opencv==4.0.1

**Note: Different versions of opencv may cause different data reading results.**

## Datasets
We use different parts of the [RESIDE](https://sites.google.com/view/reside-dehaze-datasets) dataset for training and validation.
* ITS (indoor training set)
* OTS (outdoor training set)
* SOTS (testing set)
* RTTS (real world testing samples)

Use [/code/img2path.py](https://github.com/iCVTEAM/DehazeFlow/blob/main/code/img2path.py) to read the image paths and generate path files.

## Testing
Download the trained models via https://drive.google.com/drive/folders/1Vb9BNYrDqKykfLpbX2lhNaus5YQw-s6V?usp=sharing.

Modify [DehazeFlow.yml](https://github.com/iCVTEAM/DehazeFlow/blob/main/code/DehazeFlow.yml) to:
1. set ```dataroot_GT``` and ```dataroot_HZ``` to paths containing testing images and ground-truths.
2. set ```test_mode``` to 'indoor' or 'outdoor'.
3. set ```model_path``` to pth file path.
4. set ```heat``` (standard deviation) to an appropriate value.

Run:
```
python test.py
```

## Training
Modify [DehazeFlow.yml](https://github.com/iCVTEAM/DehazeFlow/blob/main/code/DehazeFlow.yml) to:
1. set ```path_root``` to path files for training and validation.
2. set other parameters to appropriate values.

Run:
```
python -m torch.distributed.launch --nproc_per_node=2 train.py
```


## Comparison

![Comparison](https://github.com/iCVTEAM/iCVTEAM.github.io/blob/master/assets/DehazeFlow/comparison.png)


## Citation
```
@inproceedings{10.1145/3474085.3475432,
author = {Li, Hongyu and Li, Jia and Zhao, Dong and Xu, Long},
title = {DehazeFlow: Multi-Scale Conditional Flow Network for Single Image Dehazing},
year = {2021},
isbn = {9781450386517},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3474085.3475432},
doi = {10.1145/3474085.3475432},
pages = {2577–2585},
numpages = {9},
keywords = {normalizing flow, single image dehazing, attention},
location = {Virtual Event, China},
series = {MM '21}
}
```

## Acknowledgment
This repository is based on the implementation of [SRFlow: Learning the Super-Resolution Space with Normalizing Flow](https://github.com/andreas128/SRFlow).
