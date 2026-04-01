<h2 align="center"> Out of Sight but Not Out of Mind:<br>Hybrid Memory for Dynamic Video World Models </h2>

<div align="center">
    <!-- 使用 margin-bottom 负值来强行拉近与下方的距离 -->
    <div style="margin-top: -45px;margin-bottom: -55px;">
        <img src="./assets/symbol.png" alt="Project Icon" width="250">
    </div>
    <div>
        <a href="https://arxiv.org/pdf/2603.25716"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=Arxiv"></a>
        <a href="https://kj-chen666.github.io/Hybrid-Memory-in-Video-World-Models/"><img src="https://img.shields.io/badge/Homepage-project-orange.svg?logo=googlehome"></a>
        <a href="https://example.com/dataset"><img src="https://img.shields.io/badge/🤗-Dataset-yellow.svg"></a>
        <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square"></a>
    </div>
</div>

<h5 align="center"><em>
  Kaijin Chen<sup>1</sup>,
  Dingkang Liang<sup>1</sup>,
  Xin Zhou<sup>1</sup>,
  Yikang Ding<sup>2</sup>,
  Xiaoqiang Liu<sup>2</sup>,
  Pengfei Wan<sup>2</sup>,
  Xiang Bai<sup>1</sup>
</em></h5>
<div align="center">
  <sup>1</sup> Huazhong University of Science and Technology&nbsp;&nbsp;&nbsp;
  <sup>2</sup> Kling Team, Kuaishou Technology
</div>



## 🔍 Overview

While recent video world models excel at simulating static environments, they share a critical blind spot: the physical world is dynamic. When moving subjects exit the camera's field of view and later re-emerge, current models often lose track of them—rendering returning subjects as frozen statues, distorted phantoms, or letting them vanish entirely.

To bridge this gap, we introduce **Hybrid Memory**, a novel paradigm that requires models to simultaneously act as precise archivists for static backgrounds and vigilant trackers for dynamic subjects. A true world model must not only remember a subject's appearance but also mentally predict its unseen trajectory, ensuring visual and motion continuity even during out-of-view intervals.

<div align="center">    
 <img src="./assets/intro.png" width="70%" align="center" alt="Hybrid memory intro figure" />
</div>

<details>
  <summary>Abstract
  </summary>

Video world models have shown immense potential in simulating the physical world, yet existing memory mechanisms primarily treat environments as static canvases. When dynamic subjects hide out of sight and later re-emerge, current methods often struggle, leading to frozen, distorted, or vanishing subjects. We introduce **Hybrid Memory**, a novel paradigm requiring models to simultaneously act as precise archivists for static backgrounds and vigilant trackers for dynamic subjects, ensuring motion continuity during out-of-view intervals. To facilitate research in this direction, we construct **HM-World**, the first large-scale video dataset dedicated to hybrid memory. It features 59K high-fidelity clips with decoupled camera and subject trajectories, encompassing 17 diverse scenes, 49 distinct subjects, and meticulously designed exit-entry events to rigorously evaluate hybrid coherence. Furthermore, we propose **HyDRA**, a specialized memory architecture that compresses contexts into memory tokens and utilizes a spatiotemporal relevance-driven retrieval mechanism. By selectively attending to relevant motion cues, HyDRA effectively preserves the identity and motion of hidden subjects. Extensive experiments on HM-World demonstrate that our method significantly outperforms state-of-the-art approaches in both dynamic subject consistency and overall generation quality.
</details>

### 🎥 Generation Results

More results can be found on our [project homepage](https://kj-chen666.github.io/Hybrid-Memory-in-Video-World-Models/).

<div align="center">
  <img src="assets/genetation_videos/1.gif" width="32%" />
  <img src="assets/genetation_videos/2.gif" width="32%" />
  <img src="assets/genetation_videos/3.gif" width="32%" />
</div>
<div align="center">
  <img src="assets/genetation_videos/4.gif" width="32%" />
  <img src="assets/genetation_videos/5.gif" width="32%" />
  <img src="assets/genetation_videos/6.gif" width="32%" />
</div>


## 📅 TODO
* [x] Release the paper
* [ ] Release HM-World dataset
* [x] Release HyDRA checkpoints and inference code
* [x] Release HyDRA training code


## 🛠️ Installation

### Step 1: Clone this repository

```bash
git clone https://github.com/H-EmbodVis/HyDRA.git
cd HyDRA
```

> If your cloned folder name is not `HyDRA`, please `cd` into the actual folder.

### Step 2: Create & activate an Anaconda environment

```bash
conda create -n hydra python=3.10 -y
conda activate hydra
```

### Step 3: Install required packages

```bash
pip install -r requirements.txt
```

<details>
<summary><b>AMD ROCm GPU support (optional)</b></summary>

HyDRA works on AMD GPUs via ROCm. Replace Step 3 with:

```bash
# Install ROCm PyTorch (ROCm 6.4 example)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4

# Install remaining deps (skip torch/cupy)
grep -vEi "^(torch|torchvision|cupy)" requirements.txt > /tmp/req_clean.txt
pip install -r /tmp/req_clean.txt

# (Recommended) Install FlashAttention for ~16-19% faster inference
# Option A — source build with Triton backend (ROCm 6.x):
git clone https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention
cd /tmp/flash-attention && git submodule update --init third_party/aiter
FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE python setup.py install

# Option B — AMD AITER for best performance on ROCm 7.x:
#   pip install aiter  (or build from https://github.com/ROCm/aiter)
#   AITER auto-dispatches to CK kernels on ROCm 7.x (~25% faster than Triton)
```

The attention backend is selected automatically at import time:
FA3 → AITER → FA2 → SageAttention → PyTorch SDPA (fallback).

</details>

### Step 4: Download the pretrained Wan2.1 (1.3B) T2V model

- Model link: https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B
- Recommended location: `./ckpts`

Example directory structure (recommended):

```text
HyDRA/
└── ckpts/
    ├── Wan2.1_VAE.pth
    ├── diffusion_pytorch_model.safetensors
    ├── models_t5_umt5-xxl-enc-bf16.pth
    └── ... (other files)
```

### Step 5: Download the trained HyDRA weights

- Checkpoint link: https://huggingface.co/H-EmbodVis/HyDRA
- Recommended location: `./ckpts` (e.g., `./ckpts/hydra.pth`)


## 🚀 Inference

Run inference on the example data:

```bash
python infer_hydra.py
```


## 🎓 Training 

We provide an example training script that can be used to train on a custom dataset.

### Data preparation

For each training sample, please preprocess and save it as a single `.pth` file:

1. Encode the custom video into **video latents** with the VAE.
2. Encode the caption into a **text embedding** with the text encoder.
3. Convert the **camera poses** into a relative coordinate system.

Then, load these per-sample `.pth` files in the training script and train the **DiT** module.

### Train command

```bash
python train_hydra.py \
  --dit_path ./ckpts/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors \
  --use_gradient_checkpointing \
  --hydra 
```


## 👍 Acknowledgement

Thanks for the following related works and open source reposities:
* [RecamMaster](https://github.com/KlingAIResearch/ReCamMaster)
* [Context-As-Memory](https://context-as-memory.github.io)
* [DFoT](https://github.com/kwsong0113/diffusion-forcing-transformer)
* [WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay)


## 📖 Citation

If you find our work useful, please consider citing:

```bibtex
@article{chen2026out,
  title   = {Out of Sight but Not Out of Mind: Hybrid Memory for Dynamic Video World Models},
  author  = {Chen, Kaijin and Liang, Dingkang and Zhou, Xin and Ding, Yikang and Liu, Xiaoqiang and Wan, Pengfei and Bai, Xiang},
  journal = {arXiv preprint arXiv:2603.25716},
  year    = {2026}
}
