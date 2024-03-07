# [CVPR 2024] VSCode: General Visual Salient and Camouflaged Object Detection with 2D Prompt Learning
Ziyang Luo, Nian Liu, Wangbo Zhao, Xuguang Yang, Dingwen Zhang, Deng-Ping Fan, Fahad Khan, Junwei Han<br />

**Approach**: [[arxiv Paper]](https://arxiv.org/pdf/2311.15011.pdf)

<img src="https://github.com/Sssssuperior/VSCode/blob/main/task_relation.png" width="400"/><img src="https://github.com/Sssssuperior/VSCode/blob/main/network.png" width="400"/>

## 🎃 Overview
We introduce VSCode, a generalist model with novel 2D prompt learning, to jointly address four SOD tasks and three COD tasks. We utilize VST as the foundation model and introduce 2D prompts within the encoder-decoder architecture to learn domain and task-specific knowledge on two separate dimensions. A prompt discrimination loss helps disentangle peculiarities to benefit model optimization. VSCode outperforms state-of-the-art methods across six tasks on 26 datasets and exhibits zero-shot generalization to unseen tasks by combining 2D prompts, such as RGB-D COD.
<img src="https://github.com/Sssssuperior/VSCode/blob/main/method.png">

## 🎃 Environmental Setups
Pytorch $\geq$ 1.6.0, Torchvision $\geq$ 0.7.0
```
conda env create --file environment.yml -n vscode
conda activate vscode
```
## 🎃 Data Preparation
### 1. RGB SOD & RGB-D SOD
For RGB SOD and RGB-D SOD, we employ the following datasets to train our model concurrently: the training set of **DUTS** for `RGB SOD` , the training sets of **NJUD**, **NLPR**, and **DUTLF-Depth** for `RGB-D SOD`. 
For testing the RGB SOD task, we use **DUTS**, **ECSSD**, **HKU-IS**, **PASCAL-S**, **DUT-O**, and **SOD**, while **STERE**, **NJUD**, **NLPR**, **DUTLF-Depth**, **SIP**, and **ReDWeb-S** datasets are employed for testing the RGB-D SOD task. You can directly download these datasets by following [[VST]](https://github.com/nnizhang/VST?tab=readme-ov-file).

### 2. RGB-T SOD
We employ the training set of **VT5000** to train our model, and **VT821**, **VT1000**, and the testing of **VT5000** are utilized for testing (from [link](https://chenglongli.cn/code-dataset/)). Please download the corresponding contour maps from [[baidu](),PIN:] for VT5000 and place them into the `RGBT` folder.

### 3. VSOD
For VSOD, we employ six widely used benchmark datasets: **DAVIS**, **FBMS**, **ViSal**, **SegV2**, **DAVSOD-Easy**, and **DAVSOD-Normal** (from [link](https://github.com/DengPingFan/DAVSOD)). Please download corresponding contour maps from [[baidu](https://pan.baidu.com/s/1qvIduyt6A8fYlwIQJLj8Jw),PIN:o7r0] and [[baidu[(),PIN:] for DAVIS and DAVSOD, and put it into `Video` folder. For VSOD and VCOD tasks, we follow the common practice of utilizing [Flownet2.0](https://github.com/NVIDIA/flownet2-pytorch) as the optical flow extractor due to its consistently strong performance.

### 4. RGB COD
Regarding RGB COD, three extensive benchmark datasets are considered, including [**COD10K**](https://github.com/DengPingFan/SINet/), **CAMO**, and [**NC4K**](https://github.com/JingZhang617/COD-Rank-Localize-and-Segment). Please download the corresponding contour maps from [[baidu](),PIN:] for COD10K and CAMO, and put it into `COD/rgb/` folder.

### 5. VCOD
For VCOD, we utilize two widely accepted benchmark datasets: **CAD** and **MoCA-Mask** (from [link](https://github.com/XuelianCheng/SLT-Net)). Please download the corresponding contour maps from [[baidu](),PIN:] for MoCA-Mask, and put it into `COD/rgbv/` folder.

------
The total dataset folder should like this:
```
-- Data
  | -- RGB
  |    | -- DUTS
  |    | -- ECSSD

  ...

  | -- RGBD
  |    | -- NJUD
  |    | -- NLPR

  ...

  | -- RGBT
  |    | -- VT821
  |    | -- | RGB
  |    | -- | GT
  |    | -- | T
  |    | -- VT5000
  |    |    | -- Train
  |    |    | -- | RGB
  |    |    | -- | GT
  |    |    | -- | T
  |    |    | -- | Contour
  |    |    | -- Test

  ...

  | -- Video
  |    | -- Train
  |    |    | -- DAVSOD
  |    |    |    | -- select_0043
  |    |    |    | -- | RGB
  |    |    |    | -- | GT
  |    |    |    | -- | Flow
  |    |    |    | -- | Contour
  |    | -- Test
  |    |    | -- DAVIS16
  |    |    |    | -- blackswan
  |    |    |    | -- | Frame
  |    |    |    | -- | GT
  |    |    |    | -- | OF_FlowNet2

  ...

  | -- COD
  |    | -- rgb
  |    |    | -- Train
  |    |    |    | -- CAMO
  |    |    |    | -- | RGB
  |    |    |    | -- | GT
  |    |    |    | -- | Contour
  |    |    | -- Test
  |    |    |    | -- CAMO
  |    |    |    | -- | RGB
  |    |    |    | -- | GT

  ...

  |    | -- rgbv
  |    |    | -- Train
  |    |    |    | -- MoCA_Mask
  |    |    |    |    | -- TrainDataset_per_sq
  |    |    |    |    |    | -- crab
  |    |    |    |    |    | -- | Imgs
  |    |    |    |    |    | -- | GT
  |    |    |    |    |    | -- | Flow
  |    |    |    |    |    | -- | Contour
  |    |    | -- Test
  |    |    |    | -- MoCA_Mask
  |    |    |    |    |    | -- arctic_fox
  |    |    |    |    |    | -- | Imgs
  |    |    |    |    |    | -- | GT
  |    |    |    |    |    | -- | Flow

...
```

## 🎃 Experiments
Run `python train_test_eval.py --Training True --Testing True --Evaluation True` for training, testing, and evaluation which is similar to VST.

Please be aware that our evaluation tool may exhibit some differences from [Zhao Zhang](https://github.com/zzhanghub/eval-co-sod) for VSOD, as certain ground truth maps may not be binarized. 

## 🎃 Results

### 1. Model Zoo
| Name | Backbone | Params | Weight |
|  :---: |  :---:    | :---:   |  :---:   |
| VSCode-T |  Swin-T    |  54.09   |  [[baidu](https://pan.baidu.com/s/1GpGHG0z8WnzEwxi9HtnONw?pwd=97ij),PIN:97ij]   |
| VSCode-S |  Swin-S    |  74.72   |  [[baidu](https://pan.baidu.com/s/1OjRFAG7rdiUVwz3nK69Y3g?pwd=8jig),PIN:8jig]  |
| VSCode-B |  Swin-B    |  117.41  |  [[baidu](https://pan.baidu.com/s/1jbo7eu8YEE9I7KI4GqM9Rg?pwd=kidl),PIN:kidl]    |

### 2. Saliency map
Due to the dataset's size, we only offer the saliency map of **VSCode-T** [[baidu](https://pan.baidu.com/s/1Za2q554fRoUCmsAYVWveIw?pwd=ywau),PIN:ywau] and **VSCode-S** [[baidu](https://pan.baidu.com/s/19PwWRsS8woYrlJnoS2A2zA),PIN:ohf5] at this time.

## 🎃 Citation
If you use VSCode in your research or wish to refer to the baseline results published in the Model Zoo, please use the following BibTeX entry.
```
@article{luo2023vscode,
  title={VSCode: General Visual Salient and Camouflaged Object Detection with 2D Prompt Learning},
  author={Luo, Ziyang and Liu, Nian and Zhao, Wangbo and Yang, Xuguang and Zhang, Dingwen and Fan, Deng-Ping and Khan, Fahad and Han, Junwei},
  journal={arXiv preprint arXiv:2311.15011},
  year={2023}
}
```
