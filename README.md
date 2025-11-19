# [CVPR 2024] VSCode: General Visual Salient and Camouflaged Object Detection with 2D Prompt Learning & VSCode-v2: Dynamic Prompt Learning for General Visual Salient and Camouflaged Object Detection with Two-Stage Optimization
Ziyang Luo, Nian Liu, Wangbo Zhao, Xuguang Yang, Dingwen Zhang, Deng-Ping Fan, Fahad Khan, Junwei Han<br />

**Approach**: [[arxiv Paper]](https://arxiv.org/pdf/2311.15011.pdf)

<img src="https://github.com/Sssssuperior/VSCode/blob/main/task_relation.png" width="400"/><img src="https://github.com/Sssssuperior/VSCode/blob/main/network.png" width="400"/>

## 🔥 News 
The extension work of VSCode has been accepted by TPAMI. The codes, models, and results can be found in this repository. The final, camera-ready TPAMI version of the paper will be provided soon. 
The VSCode-v2 implementation is located in the **VSCode2 folder**, which contains all training, testing, and evaluation code.

## 🎃 Overview
**[VSCode]** We introduce VSCode, a generalist model with novel 2D prompt learning, to jointly address four SOD tasks and three COD tasks. We utilize VST as the foundation model and introduce 2D prompts within the encoder-decoder architecture to learn domain and task-specific knowledge on two separate dimensions. A prompt discrimination loss helps disentangle peculiarities to benefit model optimization. VSCode outperforms state-of-the-art methods across six tasks on 26 datasets and exhibits zero-shot generalization to unseen tasks by combining 2D prompts, such as RGB-D COD.

<img src="https://github.com/Sssssuperior/VSCode/blob/main/method.png">

**[VSCode-v2]** Salient object detection (SOD) and camouflaged object detection (COD) are related but distinct binary mapping tasks, each involving multiple modalities that share commonalities while maintaining unique characteristics. Existing approaches often rely on complex, task-specific architectures, leading to redundancy and limited generalization. Our previous work, VSCode, introduced a generalist model that effectively handles four SOD tasks and two COD tasks. VSCode leveraged VST as its foundation model and incorporated 2D prompts within an encoder-decoder framework to capture domain and task-specific knowledge, utilizing a prompt discrimination loss to optimize the model. Building upon the proven effectiveness of our previous work VSCode, we identify opportunities to further strengthen generalization capabilities through focused modifications in model design and optimization strategy. To unlock this potential, we propose VSCode-v2, an extension that introduces a Mixture of Prompt Experts (MoPE) layer to generate adaptive prompts. We also redesign the training process into a two stage approach: first learning shared features across tasks, then capturing specific characteristics. To preserve knowledge during this process, we incorporate distillation from our conference version model. Furthermore, we propose a contrastive learning
mechanism with data augmentation to strengthen the relationships between prompts and feature representations. VSCode-v2 demonstrates balanced performance improvements across six SOD and COD tasks. Moreover, VSCode-v2 effectively handles various multimodal inputs and exhibits zero-shot generalization
capability to novel tasks, such as RGB-D Video SOD.

<img src="https://github.com/Sssssuperior/VSCode/blob/main/vscode-v2-backbone.png">


## 🎃 Environmental Setups
Pytorch $\geq$ 1.6.0, Torchvision $\geq$ 0.7.0

## 🎃 Data Preparation
### 1. RGB SOD & RGB-D SOD
For RGB SOD and RGB-D SOD, we employ the following datasets to train our model concurrently: the training set of **DUTS** for `RGB SOD` , the training sets of **NJUD**, **NLPR**, and **DUTLF-Depth** for `RGB-D SOD`. 
For testing the RGB SOD task, we use **DUTS**, **ECSSD**, **HKU-IS**, **PASCAL-S**, **DUT-O**, and **SOD**, while **STERE**, **NJUD**, **NLPR**, **DUTLF-Depth**, **SIP**, and **ReDWeb-S** datasets are employed for testing the RGB-D SOD task. You can directly download these datasets by following [[VST]](https://github.com/nnizhang/VST?tab=readme-ov-file).

### 2. RGB-T SOD
We employ the training set of **VT5000** to train our model, and **VT821**, **VT1000**, and the testing of **VT5000** are utilized for testing (from [link](https://chenglongli.cn/code-dataset/)). Please download the corresponding contour maps from [[baidu](https://pan.baidu.com/s/18PVmR-Z2wwVtTZEr14aVJg?pwd=m9ht),PIN:m9ht] for VT5000 and place them into the `RGBT` folder.

### 3. VSOD
For VSOD, we employ six widely used benchmark datasets: **DAVIS**, **FBMS**, **ViSal**, **SegV2**, **DAVSOD-Easy**, and **DAVSOD-Normal** (from [link](https://github.com/DengPingFan/DAVSOD)). Please download corresponding contour maps and optical flow from [[baidu](https://pan.baidu.com/s/1hogrUsIEbRIWzicnI7C7Zw?pwd=jyzy),PIN:jyzy] and [[baidu[(https://pan.baidu.com/s/1IUPH8jG-t2ZlK1Acw1W1oA),PIN:bxi7] for DAVIS and DAVSOD, and put it into `Video` folder. For VSOD and VCOD tasks, we follow the common practice of utilizing [Flownet2.0](https://github.com/NVIDIA/flownet2-pytorch) as the optical flow extractor due to its consistently strong performance.

### 4. RGB COD
Regarding RGB COD, three extensive benchmark datasets are considered, including [**COD10K**](https://github.com/DengPingFan/SINet/), **CAMO**, and [**NC4K**](https://github.com/JingZhang617/COD-Rank-Localize-and-Segment). Please download the corresponding contour maps from [[baidu](https://pan.baidu.com/s/1guTrF3maDesAlG0t0utdGw),PIN:gkq2] and [[baidu](https://pan.baidu.com/s/1QRqqv7IMicudRW0MwnI8FA),PIN:zojp] for COD10K and CAMO, and put it into `COD/rgb/` folder.

### 5. VCOD
For VCOD, we utilize two widely accepted benchmark datasets: **CAD** and **MoCA-Mask** (from [link](https://github.com/XuelianCheng/SLT-Net)). Please download the corresponding contour maps and optical flow from [[baidu](https://pan.baidu.com/s/1RGxO8HQWct9ybVr_CZDtxw),PIN:tjah] for MoCA-Mask, and put it into `COD/rgbv/` folder.

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

### 6. Augmentation Dataset
For VSCode-v2, we introduce concatenated augmentation data for the contrastive loss. We recommend that readers generate their own datasets as well. For reference, we list the datasets we generated, including **RGB_pseudo** [[baidu](https://pan.baidu.com/s/1s22_s_6mov7MlBVDIPfwjg?pwd=j7q5),PIN:j7q5], **RGBD_pseudo** [[baidu](https://pan.baidu.com/s/1IXDWtF3-TSbwb2OtDon3AQ?pwd=3k1p),PIN:3k1p], **RGBT_pseudo** [[baidu](https://pan.baidu.com/s/1-_sYGUu_ZiItt76R0aGHlw?pwd=kj8b),PIN:kj8b], **RGBV_pseudo** [[baidu](https://pan.baidu.com/s/1XgKMjj_Rdnj8ZVQQIsp22A?pwd=kprh),PIN:kprh], **CODRGB_pseudo** [[baidu](https://pan.baidu.com/s/1T3iMZvXiMbDg1__z-S7Qfg?pwd=2if2),PIN:2if2] and **CODRGBV_pseudo** [[baidu](https://pan.baidu.com/s/1t7HqlWR_caOeaT8Hhe4KZw?pwd=di5c),PIN:di5c].


## 🎃 Experiments
Run `python train_test_eval.py --Training True --Testing True --Evaluation True` for training, testing, and evaluation which is similar to VST.

Please be aware that our evaluation tool may exhibit some differences from [Zhao Zhang](https://github.com/zzhanghub/eval-co-sod) for VSOD, as certain ground truth maps may not be binarized. 

## 🎃 Results
**Due to the limited storage capacity of my Google Drive, I am unable to upload additional files there. If you can only access the data via Google Drive and are unable to use Baidu Cloud, please contact me by email. (ziyangluo1110@gmail.com).**
### 1. Model Zoo
| Name | Backbone | Params | Weight |
|  :---: |  :---:    | :---:   |  :---:   |
| VSCode-T |  Swin-T    |  54.09   |  [[baidu](https://pan.baidu.com/s/11jFy0liVMStQdeEtaVxy-A),PIN:mmn1]/[[Geogle Drive](https://drive.google.com/file/d/1znYlazhiG2vit113MAE5bH5erDf6Ru2_/view?usp=sharing)]|
| VSCode-S |  Swin-S    |  74.72   |  [[baidu](https://pan.baidu.com/s/1OjRFAG7rdiUVwz3nK69Y3g?pwd=8jig),PIN:8jig]/[[Geogle Drive](https://drive.google.com/file/d/1rqUE7fh5CO34_ypM3gggNM7tdUqeZUxH/view?usp=sharing)]|
| VSCode-B |  Swin-B    |  117.41  |  [[baidu](https://pan.baidu.com/s/1jbo7eu8YEE9I7KI4GqM9Rg?pwd=kidl),PIN:kidl]/[[Geogle Drive](https://drive.google.com/file/d/1zKJnH1ZY08L2ul5mPQocbTSHeale6JWE/view?usp=drive_link)]|
| VSCode-v2-T(stage1) |  Swin-T    |  -  |  [[baidu](https://pan.baidu.com/s/1OOKz9YRYlhceCGFO4a4Jyg?pwd=wexs),PIN:wexs]|
| VSCode-v2-S(stage1) |  Swin-S    |  -  |  [[baidu](https://pan.baidu.com/s/1nGR29uUKQU2wURebSF9LDA?pwd=gnma),PIN:gnma]|
| VSCode-v2-T(stage2) |  Swin-T    |  69.8  |  [[baidu](https://pan.baidu.com/s/1U5vrfw9CFwQevHuOaEA8cg?pwd=8imx),PIN:8imx]|
| VSCode-v2-S(stage2) |  Swin-S    |  90.4  |  [[baidu](https://pan.baidu.com/s/1T6l3UL7yQvtnDRs40quKwQ?pwd=4r7b),PIN:4r7b]|


### 2. Prediction Maps
We offer the prediction maps of **VSCode-T** [[baidu](https://pan.baidu.com/s/13MKOObYH6afYzF7P-2vjeQ),PIN:gsvf]/ [[Geogle Drive](https://drive.google.com/file/d/1paABoJ_Tx4uV1XAw6o-QE05b-c1M4o3V/view?usp=drive_link)] , **VSCode-S** [[baidu](https://pan.baidu.com/s/19PwWRsS8woYrlJnoS2A2zA),PIN:ohf5]/[[Geogle Drive](https://drive.google.com/file/d/1uZfzCePoRXgqQso80mR0bCFH2_7dZ1g_/view?usp=drive_link)], **VSCode-B** [[baidu](https://pan.baidu.com/s/1M1TsyvzPriCFyY8-QlWbjA),PIN:uldc]/[[Geogle Drive](https://drive.google.com/file/d/1vu_mu93p2rczLzvq4yx29lneCc0t_wOz/view?usp=sharing)], **VSCode-v2-T** [[baidu](https://pan.baidu.com/s/1lVkHvV1BcFP_0iQ_-0acVQ?pwd=x787),PIN:x787]] , **VSCode-v2-S** [[baidu](https://pan.baidu.com/s/1Pqb2NFmZVPUdLYN_I6IqPQ?pwd=v2i6),PIN:v2i6]] at this time.

## 🎃 Citation
If you use VSCode or VSCode-v2 in your research or wish to refer to the baseline results published in the Model Zoo, please use the following BibTeX entry.
```
@inproceedings{luo2024vscode,
  title={Vscode: General visual salient and camouflaged object detection with 2d prompt learning},
  author={Luo, Ziyang and Liu, Nian and Zhao, Wangbo and Yang, Xuguang and Zhang, Dingwen and Fan, Deng-Ping and Khan, Fahad and Han, Junwei},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={17169--17180},
  year={2024}
}
```
