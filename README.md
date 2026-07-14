<!-- markdownlint-disable MD033 MD041 -->
<div align="center">

<img src="assets/meanflownft_logo.jpg" width="48%" alt="MeanFlowNFT logo">

<h2>Bringing Forward-Process RL to Average-Velocity Generators</h2>

[Yushi Huang](https://harahan.github.io/)<sup>1,2</sup>\* ·
[Xiangxin Zhou](https://zhouxiangxin1998.github.io/)<sup>1</sup>\*<sup>✉</sup> ·
[Jun Zhang](https://eejzhang.people.ust.hk/)<sup>2</sup> ·
[Liefeng Bo](https://research.cs.washington.edu/istc/lfb/)<sup>1</sup> ·
[Tianyu Pang](https://p2333.github.io/)<sup>1,✉</sup>

<sup>1</sup>Tencent Hunyuan &nbsp;&nbsp; <sup>2</sup>The Hong Kong University of Science and Technology

\* Equal contribution &nbsp;&nbsp; <sup>✉</sup>Corresponding authors

[![arXiv](https://img.shields.io/badge/arXiv-xxx-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/xxx)
[![Project Page](https://img.shields.io/badge/Project-Page-0F9D8A?style=for-the-badge&logo=googlechrome&logoColor=white)](https://harahan.github.io/meanflownft-project-page/)
[![Hugging Face](https://img.shields.io/badge/🤗-Models-FFD21E?style=for-the-badge)](https://huggingface.co/Harahan/MeanFlowNFT)

[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

</div>

> [!IMPORTANT]
> This is the **Wan2.1 branch**. The SD3.5 implementation remains on
> [`main`](https://github.com/Harahan/MeanFlowNFT/tree/main).
> Wan2.1 support on this branch is still under active development.

- 🚀 **Forward-process RL for MeanFlow:** optimize rewards in induced
  instantaneous-velocity space while keeping the model in average-velocity
  space.
- ⚡ **Native any-step generation:** preserve MeanFlow's efficient sampler
  without reverse-trajectory backpropagation or likelihood estimation.
- 📈 **Policy improvement:** inherit DiffusionNFT's strict improvement
  guarantee under the idealized pointwise optimum.

<p align="center">
  <a href="https://harahan.github.io/meanflownft-project-page/"><img src="assets/teaser.webp" width="100%" alt="Selected MeanFlowNFT image and video generations"></a>
</p>

See the [project page](https://harahan.github.io/meanflownft-project-page/)
to play the videos and explore the full interactive comparison.

---

## 🍭 Overview

<table align="center">
  <tr>
    <td align="center" width="58%" valign="middle">
      <img src="assets/overview.webp" width="100%" alt="MeanFlowNFT method overview">
      <br><sub>Induced instantaneous-velocity optimization with native MeanFlow sampling.</sub>
    </td>
    <td align="center" width="42%" valign="middle">
      <img src="assets/algorithm.jpg" width="100%" alt="MeanFlowNFT update algorithm">
      <br><sub>One practical MeanFlowNFT update step.</sub>
    </td>
  </tr>
</table>

- 🧭 **Optimize in instantaneous-velocity space** through the MeanFlow
  identity while retaining the average-velocity parameterization.
- 🔁 **Use a shared EMA finite-difference derivative** for stable positive and
  negative policy construction.
- 🪶 **Train only from forward-noised clean samples;** inference still calls
  the original MeanFlow sampler.

---

## 📊 Results

<p align="center">
  <a href="assets/results_sd35.png"><img src="assets/results_sd35.png" width="100%" alt="SD3.5-Medium image-generation results"></a>
</p>

<p align="center">
  <a href="assets/results_wan.png"><img src="assets/results_wan.png" width="100%" alt="Wan2.1 video-generation results"></a>
</p>

MeanFlowNFT is best on **6 of 8** image metrics among evaluated few-step
models, while 4-step Wan2.1 reaches **84.33 VBench**, surpassing 50-step
LongCat-Video RL. See the
[project page](https://harahan.github.io/meanflownft-project-page/) for full
scaling curves and qualitative comparisons.

---

## 🛠️ Installation

The Wan paper environment uses Python 3.10, PyTorch 2.6.0, and CUDA 12.4.

```bash
git clone --branch wan https://github.com/Harahan/MeanFlowNFT.git
cd MeanFlowNFT

conda create -n meanflownft-wan python=3.10 -y
conda activate meanflownft-wan
pip install -r requirements.txt
pip install hpsv3==1.0.0 --no-deps
pip install trl==0.12.2 --no-deps
pip install -e .
```

HPSv3 and VideoAlign were authored against older Transformers releases. This
branch pins the paper environment to Transformers 4.57 and applies scoped
checkpoint/import compatibility fixes without modifying installed packages.

### 📦 Prepare Wan checkpoints

MeanFlowNFT training starts from the full
[`nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers`](https://huggingface.co/nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers)
pipeline. Normal Wan inference uses
[`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers).
The loaders accept either these Hub IDs or local Diffusers directories.
For shared multi-node training, download the AnyFlow pipeline once:

```bash
hf auth login
hf download nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers \
    --local-dir models/AnyFlow-Wan2.1-T2V-1.3B-Diffusers
export MEANFLOWNFT_ANYFLOW_WAN_PATH="$(pwd)/models/AnyFlow-Wan2.1-T2V-1.3B-Diffusers"
```

> [!CAUTION]
> The AnyFlow-Wan model license restricts the model and derivative weights to
> non-commercial use. Review its license before training or distributing
> checkpoints.

## 🎁 Video Reward Preparation

Wan MeanFlowNFT uses HPSv3 general/percentile rewards and VideoAlign MQ/TA.
HPSv3 downloads its model through Hugging Face when first used. Install
VideoAlign and its VideoReward checkpoint outside the Python package:

```bash
git clone https://github.com/ModelTC/VideoAlign.git third_party/VideoAlign
git -C third_party/VideoAlign checkout aba26b658fec7d9fd30c295187b548ea673c8769

hf download KwaiVGI/VideoReward --local-dir models/VideoReward
```

The committed training config points to `third_party/VideoAlign` and
`models/VideoReward`. Override either path without editing the YAML:

```bash
--override \
meanflow_nft.reward_model_paths.videoalign_dir=/path/to/VideoAlign \
meanflow_nft.reward_model_paths.videoalign_ckpt=/path/to/VideoReward
```

`imageio-ffmpeg` supplies the MP4 encoder used for rewards, previews, and
inference outputs.

## 🚀 Training

This branch fine-tunes an already distilled AnyFlow-Wan policy; AnyFlow
pretraining and on-policy distillation are not repeated here. The
`dataset/wan_dancegrpo/` training and evaluation prompt lists are included
directly in this repository.

The canonical run uses four nodes with eight GPUs each. On every node, set the
standard distributed variables, changing only `NODE_RANK`:

```bash
export NNODES=4
export NODE_RANK=0
export MASTER_ADDR=10.0.0.1
export MASTER_PORT=29500

bash scripts/train_wan_meanflow_nft.sh \
    8 configs/meanflow_nft/wan2.1_t2v_1.3b_meanflow_nft.yaml \
    --override \
    model.pretrained_path="${MEANFLOWNFT_ANYFLOW_WAN_PATH}"
```

The launcher contains no environment activation, credentials, proxy settings,
or cluster-specific network interface. Set site-specific NCCL variables in the
job environment when required.

To resume, append:

```bash
train.resume_from=/path/to/checkpoint-<epoch>
```

Checkpoints include the LoRA tensors and the fully trained
`condition_embedder.delta_embedder`. For evaluation/inference, prefer
`generator_ema.pt` or the checkpoint's exported `transformer/` adapter.

## 🔮 Inference and Evaluation

The final policy first loads the complete AnyFlow-Wan pipeline, then injects
the configured LoRA modules and applies both NFT adapter components: all LoRA
A/B tensors and the complete `delta_embedder` state.

```bash
export MEANFLOWNFT_WAN_CKPT=/path/to/checkpoint/generator_ema.pt
export NUM_GPUS=1
export NUM_STEPS=4

bash scripts/inference_wan_meanflow_nft_steps.sh \
    configs/inference/wan2.1_t2v_1.3b.yaml "${NUM_GPUS}" "${NUM_STEPS}"
```

Generated videos are written to
`inference_outputs/wan_meanflow_nft/steps_4/meanflow_nft/`. Passing a
conventional LoRA without `delta_embedder` is rejected rather than silently
producing the AnyFlow baseline.

To evaluate the final EMA policy on the full
`dataset/wan_dancegrpo/test.txt` set with HPSv3 general, HPSv3 percentile,
VideoAlign MQ, and VideoAlign TA, and then run VBench in the same evaluation:

```bash
bash scripts/train_wan_meanflow_nft.sh \
    8 configs/meanflow_nft/wan2.1_t2v_1.3b_meanflow_nft.yaml \
    --eval-only \
    --override \
    model.pretrained_path="${MEANFLOWNFT_ANYFLOW_WAN_PATH}" \
    train.resume_from=/path/to/checkpoint-1600 \
    eval.vbench.enabled=true \
    eval.vbench.cache_dir=/path/to/vbench/model/cache
```

The AnyFlow base is loaded from `model.pretrained_path` (overridden above),
while `train.resume_from` points to the complete MeanFlowNFT checkpoint
directory.
The VBench evaluator and both official 946-prompt metadata files are included
under `meanflownft/eval/` and `dataset/vbench/`; no external AnyFlow checkout
is required. Generation and the 16 metric dimensions are distributed across
all ranks before quality, semantic, and overall scores are aggregated.

## 📄 Citation

```bibtex
@misc{huang2026meanflownft,
  title  = {MeanFlowNFT: Bringing Forward-Process RL to Average-Velocity Generators},
  author = {Huang, Yushi and Zhou, Xiangxin and Zhang, Jun and Bo, Liefeng and Pang, Tianyu},
  year   = {2026},
  note   = {Preprint}
}
```

## 🙌 Acknowledgements

This implementation builds on
[MeanFlow](https://github.com/Gsunshine/meanflow),
[AnyFlow](https://github.com/NVlabs/AnyFlow),
[DiffusionNFT](https://github.com/NVlabs/DiffusionNFT),
[VideoAlign](https://github.com/ModelTC/VideoAlign),
[VBench](https://github.com/Vchitect/VBench),
[Diffusers](https://github.com/huggingface/diffusers),
[Transformers](https://github.com/huggingface/transformers), and
[PEFT](https://github.com/huggingface/peft).

## ⚖️ License

This project is released under the [Apache License 2.0](LICENSE). Model and
reward checkpoints remain subject to their original licenses and terms.
