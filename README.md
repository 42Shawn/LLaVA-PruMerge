# LLaVA-PruMerge: Adaptive Token Reduction for Efficient Large Multimodal Models

[Yuzhang Shang](https://42shawn.github.io/)\*, [Mu Cai](https://pages.cs.wisc.edu/~mucai/)\*, Bingxin Xu, [Yong Jae Lee](https://pages.cs.wisc.edu/~yongjaelee/)^, [Yan Yan](https://tomyan555.github.io/)^

\*Equal Contribution, ^Equal Advising

[[Paper](https://arxiv.org/abs/2403.15388)] [[Project Page](https://llava-prumerge.github.io/)]

<div align="center">
  <img src="https://llava-prumerge.github.io/images/architecture.png" alt="Our approach" width="50%">
</div>


## How to run.
### Step.0: Set the environment the same as LLaVA-1.5

### Step.1 (for inference): Download Checkpoints
Download the checkpoint from [here]([https://arxiv.org/abs/2310.00034](https://drive.google.com/file/d/1hGc6Pet_GbkwUoKMbXhr5PS7kiIE__eT/view?usp=sharing)https://drive.google.com/file/d/1hGc6Pet_GbkwUoKMbXhr5PS7kiIE__eT/view?usp=sharing) (LoRA Version) to checkpoints/llava-v1.5-7b-lora-prunemerge.

### Step.2 (for inference): Run the script.
```shell
CUDA_VISIBLE_DEVICES=7 XDG_CACHE_HOME='/data/shangyuzhang/' bash scripts/v1_5/eval/testvqa.sh
```
