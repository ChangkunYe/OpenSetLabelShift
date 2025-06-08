# Open Set Label Shift with Test Time Out-of-Distribution Reference
This is the official implementation for CVPR 2025 paper "Open Set Label Shift with Test Time Out-of-Distribution Reference". Paper link is [here](https://openaccess.thecvf.com/content/CVPR2025/papers/Ye_Open_Set_Label_Shift_with_Test_Time_Out-of-Distribution_Reference_CVPR_2025_paper.pdf).

The project is established largely on the [OpenOOD 1.5 project](https://github.com/Jingkang50/OpenOOD) publicly available.
If you find this repository useful or use this code in your research, please cite the following paper(s): 
 ```
@inproceedings{ye2025open,
  title={Open Set Label Shift with Test Time Out-of-Distribution Reference},
  author={Ye, Changkun and Tsuchida, Russell and Petersson, Lars and Barnes, Nick},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={30619--30629},
  year={2025}
}
 ```
as well as the OpenOOD paper:
```
@article{zhang2023openood,
  title={OpenOOD v1.5: Enhanced Benchmark for Out-of-Distribution Detection},
  author={Zhang, Jingyang and Yang, Jingkang and Wang, Pengyun and Wang, Haoqi and Lin, Yueqian and Zhang, Haoran and Sun, Yiyou and Du, Xuefeng and Li, Yixuan and Liu, Ziwei and Chen, Yiran and Li, Hai},
  journal={arXiv preprint arXiv:2306.09301},
  year={2023}
}
```
and other OOD detection papers if you are comparing with.

If you are only interest in the proposed Open Set Label Shift estimation and correction algorithm. The implementation is provided in "OpenSetLabelShift/openood/label_shift/".
## Requirements

### Installation

Please follow the instruction in README-openood.md for the OpenOOD project requirement, which largely covers the requirement of this project.

Special Requirement:
For the label shift estimation models tested, [Cvxpy](https://www.cvxpy.org/) package is used in the closed set label shift BBSE method. Which can be installed via conda:
```
conda install -c conda-forge cvxpy
```
### Dataset Preparation
Please use bash scripts in "./script/download/" to download the dataset and checkpoints.

## Train In-Distribution (ID) Classifiers
Please adjust the bash script in "./script/basics/$DATASET/train-**.sh" and train your desired classifier
When train the Neural Network classifier from scratch, the recommended hardware setup is as follows:

|   Dataset    |    #GPU    |        #CPU         |
|:------------:|:----------:|:-------------------:|
| CIFAR10/100  | &ge; 2 Gb  | &gt; 4 + 1 threads  |
| ImageNet-200 | &ge; 12 Gb | &gt; 16 + 1 threads |

One Nvidia RTX 2080Ti is sufficient for the training. But it's better to have at least 16 Gb GPU space if you want to test ImageNet 1k.


## Test Label Shift Estimation Model

To test existing models' performance under label shift, adjust the dataset path "$data_path" and checkpoint path "$ckpt_path" in the bash script "./test_OSLS.sh" and run: 
```
bash ./test_OSLS.sh
```
The performance will be printed. Import performance will be recorded in "./results/ood/".


## License
Please see LICENSE and LICENSE-openood

## Questions
Please raise issues or contact author at changkun.ye@anu.edu.au.

P.S. Please also consider checking issues in the OpenOOD project at https://github.com/Jingkang50/OpenOOD.
