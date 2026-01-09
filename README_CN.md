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


## Introduction
图像美学评估（IAA）通过以用户为中心的感知分析来评价视觉质量，并能够指导多种应用。多模态大语言模型（MLLMs）的最新进展引发了人们对其应用于IAA的兴趣。然而，将MLLMs应用于IAA时仍存在两个关键局限：

1) tokenizer策略导致对分数不敏感；
2) 基于分类的解码机制引入了分数量化误差。
3) 目前基于MLLM的IAA方法将任务视为粗粒度评级分类，再通过概率到分数的映射完成评估，这丢失了细粒度信息。

为应对这些挑战，我们提出了ROC4MLLM，从两个角度提供互补解决方案：

1) 表征层面：我们将分数从词词元空间中分离，避免将分数作为文本进行tokenize。通过独立的位置token桥接这两个空间，提升模型对文本中分数位置的敏感性。
2) 损失计算层面：我们对文本预测和分数预测采用不同的损失函数，增强模型对分数梯度的敏感性。将分数与文本解耦，既确保了有效的监督，又防止了损失计算中分数与文本相互干扰。

在五个数据集上的大量实验证明，ROC4MLLM无需额外训练数据即可实现SOTA性能。此外，其即插即用的设计确保了与现有MLLMs的无缝集成，有效提升其IAA性能。
<img alt="method" src="https://github.com/user-attachments/assets/4e6e7509-6679-4fae-a05d-4191caff42a6" />


## 权重
* 权重下载地址 [Baidu Netdisk](https://pan.baidu.com/s/1GwX6AEsJ3txDpxPeCdY6LA?pwd=bupt). 
* 直接可运行的Docker环境 [Baidu Netdisk](https://pan.baidu.com/s/1IjRsT691hom9naxtjlIzKw?pwd=bupt).  

## 使用

### 安装
1. 克隆此仓库并进入 ROC4MLLM 文件夹。
```bash
git clone https://github.com/woshidandan/Assessing-Image-Aesthetics-via-Multimodal-Large-Language-Models.git
cd ROC4MLLM
```

2. 安装环境
```Shell
conda create -n roc4mllm python=3.10 -y
conda activate roc4mllm
pip install --upgrade pip
pip install -e .
```

3. 安装训练所需要的包
```
pip install -e ".[train]"
pip install flash-attn --no-buil
```

### 快速开始
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

## 训练
### 准备训练数据
数据准备请参考 [mPLUG-Owl2](https://github.com/X-PLUG/mPLUG-Owl)。

**请注意**： 我们已添加 `gt_score` 字段。如果您计划使用 CE 损失或 EMD 损失，则数据中必须包含 `target` 字段。

以下是 AVA 数据集中的一个数据示例：

```python
{
  "image": "771257.jpg",
  "gt_score": 3.463414634146341,
  "conversations": [{"from": "human", "value": "<|image|>Could you evaluate the aesthetics of this image?"}, {"from": "gpt", "value": "The aesthetic rate of the image is [SCORE]. "}],
  "target": [0.15853658536585366, 0.10975609756097561, 0.2073170731707317, 0.2926829268292683, 0.16463414634146342, 0.04878048780487805, 0.0, 0.0, 0.006097560975609756, 0.012195121951219513]
}
```
请将您的数据文件路径填入 `scripts/finetune.sh` 中的 `DATA_FILE`。同时，需要更新同一脚本中的 `Image_root`，使其指向存储原始图像的目录。

### 准备模型权重
请下载预训练模型权重，并相应更新 `scripts/finetune.sh` 中的 `LOAD`。
### 训练脚本
运行以下命令开始训练：
```
bash scripts/finetune.sh
```
您可以修改 `min_score` 和 `max_score` 以匹配您数据集中分数的取值范围。通过 `l1_weight`、`ce_weight` 和 `emd_weight` 参数，确定分数损失所使用的损失函数及其权重。

**重要提示**： 如果使用 CE 或 EMD 损失，请确保 `num_tokens` 参数与您训练数据中 `target` 字段的长度保持一致。



## 如果你觉得我们的工作对你有帮助，欢迎引用:
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
