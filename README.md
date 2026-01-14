[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)

<div align="center">
<h1>
<b>
Regression Over Classification: Assessing Image Aesthetics via Multimodal Large Language Models
</b>
</h1>
<h4>
<b>
Xingyuan Ma, Shuai He, Anlong Ming, Haobin Zhong, Huadong Ma
    
Beijing University of Posts and Telecommunications
</b>
</h4>
</div>

-----------------------------------------


[国内的小伙伴可以看这](https://github.com/woshidandan/Assessing-Image-Aesthetics-via-Multimodal-Large-Language-Models/blob/711414fd28f1377fc6bbe5a4ea8d6d79a64c429b/README_CN.md)
This repo contains the official implementation of the **AAAI 2026** paper.

## Introduction
Image Aesthetics Assessment (IAA) evaluates visual quality through user-centered perceptual analysis and can guide various applications. Recent advances in Multimodal Large Language Models (MLLMs) have sparked interest in adapting them for IAA. However, two critical limitations persist in applying MLLMs to IAA: 
  1) the tokenization strategy leads to insensitivity to scores
  2) the classification-based decoding mechanisms introduce score quantization errors
  3) Current MLLM-based IAA methods treat the task as coarse rating classification followed by probability-to-score mapping, which loses fine-grained information. 

To address these challenges, we propose ROC4MLLM, offering complementary solutions from two perspectives:
  1) Representation: We separate scores from the word token space to avoid tokenizing scores as text. An independent position token bridges these spaces, improving the model’s sensitivity to score positions in text.
  2) Computation: We apply distinct loss functions for text and score predictions to enhance the model’s sensitivity to score gradients. Decoupling scores from text ensures effective supervision while preventing interference between scores and text in the loss computation.

Extensive experiments across five datasets demonstrate ROC4MLLM’s state-of-the-art performance without requiring additional training data. Additionally, ROC4MLLM’s plug-and-play design ensures seamless integration with existing MLLMs, boosting their IAA performance.

<img alt="method" src="https://github.com/user-attachments/assets/4e6e7509-6679-4fae-a05d-4191caff42a6" />


## Checkpoints
* Download the weight from [Baidu Netdisk](https://pan.baidu.com/s/1GwX6AEsJ3txDpxPeCdY6LA?pwd=bupt) or [HuggingFace](https://huggingface.co/Ricardo-M/ROC4MLLM). 
* Or Download the Docker environment from [Baidu Netdisk](https://pan.baidu.com/s/1IjRsT691hom9naxtjlIzKw?pwd=bupt). 

## Usage

### Install
1. Clone this repository and navigate to ROC4MLLM folder
```bash
git clone https://github.com/woshidandan/Assessing-Image-Aesthetics-via-Multimodal-Large-Language-Models.git
cd ROC4MLLM
```

2. Install Package
```Shell
conda create -n roc4mllm python=3.10 -y
conda activate roc4mllm
pip install --upgrade pip
pip install -e .
```

3. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build
```

### Quick Start Code
```python
from mplug_owl2.assessor import Assessment
from PIL import Image

assessment=Assessment(pretrained="../ROC4MLLM_weights")
images=["test_images/1_-10.jpg","test_images/1_-10.jpg"]
input_img=[]
for image in images:
    img=Image.open(image).convert('RGB')
    input_img.append(img)
answer=assessment(input_img,precision=4)
print(answer)
```

## Training
### Prepare Training Data
Please refer to [mPLUG-Owl2](https://github.com/X-PLUG/mPLUG-Owl) for data preparation.

**Notes**: We have added a `gt_score` field. If you intend to use CE loss or EMD loss, the `target` field is required.

Below is an example of a data sample in AVA:
```python
{
  "image": "771257.jpg",
  "gt_score": 3.463414634146341,
  "conversations": [{"from": "human", "value": "<|image|>Could you evaluate the aesthetics of this image?"}, {"from": "gpt", "value": "The aesthetic rate of the image is [SCORE]. "}],
  "target": [0.15853658536585366, 0.10975609756097561, 0.2073170731707317, 0.2926829268292683, 0.16463414634146342, 0.04878048780487805, 0.0, 0.0, 0.006097560975609756, 0.012195121951219513]
}
```
Place your data file path in the `DATA_FILE` within `scripts/finetune.sh`. You also need to update the `Image_root` in the same script to point to the directory where your original images are stored.

### Prepare model checkpoint
Download the pretrained model checkpoints and update the `LOAD` in `scripts/finetune.sh` accordingly.
### Training scripts
Run the following command to start training:
```
bash scripts/finetune.sh
```
You can modify `min_score` and `max_score` to define the score range in your dataset. Use `l1_weight`, `ce_weight`, and `emd_weight` to configure the loss functions and their respective weights for the score loss.

**Important Note**: If you use CE or EMD loss, ensure that the `num_tokens` matches the length of the `target` field in your training data.



## If you find our work is useful, pleaes cite our paper:
```
@inproceedings{MaRegression,
  title     = {Regression Over Classification: Assessing Image Aesthetics via Multimodal Large Language Models},
  author    = {Ma, Xingyuan and He, Shuai and Ming, Anlong and Zhong, Haobin and Ma, Huadong},
  booktitle = {Proceedings of the 40th AAAI Conference on Artificial Intelligence (AAAI)},
  year      = {2026}
}
```
<table style="width:100%; border-collapse:collapse; text-align:center;">
  <thead>
    <tr>
      <th>🎁 Projects</th>
      <th>📚 Publication</th>
      <th>🌈 Content</th>
      <th>⭐ Stars</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://github.com/woshidandan/Assessing-Image-Aesthetics-via-Multimodal-Large-Language-Models"><strong>Assessing Aesthetics via MLLMs【解决MLLMs对分数不敏感问题】</strong></a></td>
      <td><strong>AAAI 2026 (Oral) </strong></td>
      <td><strong>Code</strong></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/Assessing-Image-Aesthetics-via-Multimodal-Large-Language-Models?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/chasecjg/ICTA2Net"><strong>Aesthetics of Image Color Temperature【色温美学评估】</strong></a></td>
      <td><strong>AAAI 2026</strong></td>
      <td><strong>Code, Dataset</strong></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/chasecjg/ICTA2Net?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/Attacker-against-image-aesthetics-assessment-model"><strong>Attacker Against IAA Model【美学模型的攻击和安全评估框架】</strong></a></td>
      <td><strong>TIP 2025</strong></td>
      <td><strong>Code, Dataset</strong></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/Attacker-against-image-aesthetics-assessment-model?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/Rethinking-Personalized-Aesthetics-Assessment"><strong>Personalized Aesthetics Assessment【个性化美学评估新范式】</strong></a></td>
      <td><strong>CVPR 2025 (Highlight)</strong></td>
      <td><strong>Code, Dataset</strong></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/Rethinking-Personalized-Aesthetics-Assessment?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/Pixel-level-No-reference-Image-Exposure-Assessment"><strong>Pixel-level image exposure assessment【首个像素级曝光评估】</strong></a></td>
      <td><strong>NIPS 2024</strong></td>
      <td><strong>Code, Dataset</strong></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/Pixel-level-No-reference-Image-Exposure-Assessment?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/Long-Tail-image-aesthetics-and-quality-assessment"><strong>Long-tail solution for image aesthetics assessment【美学评估数据不平衡解决方案】</strong></a></td>
      <td><strong>ICML 2024</strong></td>
      <td><strong>Code</strong></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/Long-Tail-image-aesthetics-and-quality-assessment?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/Prompt-DeT"><strong>CLIP-based image aesthetics assessment【基于CLIP多因素色彩美学评估】</strong></a></td>
      <td><strong>Information Fusion 2024</strong></td>
      <td><strong>Code, Dataset</strong></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/Prompt-DeT?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/SR-IAA-image-aesthetics-and-quality-assessment"><strong>Compare-based image aesthetics assessment【基于对比学习的多因素美学评估】</strong></a></td>
      <td><strong>ACMMM 2024</strong></td>
      <td><strong>Code</strong></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/SR-IAA-image-aesthetics-and-quality-assessment?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/Image-Color-Aesthetics-and-Quality-Assessment"><strong>Image color aesthetics assessment【首个色彩美学评估】</strong></a></td>
      <td><strong>ICCV 2023</strong></td>
      <td><strong>Code, Dataset</strong></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/Image-Color-Aesthetics-and-Quality-Assessment?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/Image-Aesthetics-and-Quality-Assessment"><strong>Image aesthetics assessment【通用美学评估】</strong></a></td>
      <td><strong>ACMMM 2023</strong></td>
      <td><strong>Code</strong></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/Image-Aesthetics-and-Quality-Assessment?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/TANet-image-aesthetics-and-quality-assessment"><strong>Theme-oriented image aesthetics assessment【首个多主题美学评估】</strong></a></td>
      <td><strong>IJCAI 2022</strong></td>
      <td><strong>Code, Dataset</strong></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/TANet-image-aesthetics-and-quality-assessment?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/AK4Prompts"><strong>Select prompt based on image aesthetics assessment【基于美学评估的提示词筛选】</strong></a></td>
      <td><strong>IJCAI 2024</strong></td>
      <td><strong>Code</strong></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/AK4Prompts?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/mRobotit/M2Beats"><strong>Motion rhythm synchronization with beats【动作与韵律对齐】</strong></a></td>
      <td><strong>IJCAI 2024</strong></td>
      <td><strong>Code, Dataset</strong></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/mRobotit/M2Beats?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://github.com/woshidandan/Champion-Solution-for-CVPR-NTIRE-2024-Quality-Assessment-on-AIGC"><strong>Champion Solution for AIGC Image Quality Assessment【NTIRE AIGC图像质量评估赛道冠军】</strong></a></td>
      <td><strong>CVPRW NTIRE 2024</strong></td>
      <td><strong>Code</strong></td>
      <td><img alt="Stars" src="https://img.shields.io/github/stars/woshidandan/Champion-Solution-for-CVPR-NTIRE-2024-Quality-Assessment-on-AIGC?style=flat-square&labelColor=343b41"/></td>
    </tr>
    <tr>
      <td><a href="https://www.itu.int/ITU-T/recommendations/rec.aspx?id=15614"><strong>ITU Standard for Image Aesthetics Assessment Metadata【首个图像美学元数据国际标准】</strong></a></td>
      <td><strong>ITU 2023</strong></td>
      <td><strong>-</strong></td>
      <td><strong>-</strong></td>
    </tr>
  </tbody>
</table>
