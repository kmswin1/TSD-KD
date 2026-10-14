# TSD-KD
Explain in Your Own Words: Improving Reasoning via Token-Selective Dual Knowledge Distillation

#### Training from UltraInteract_sft [TSD-KD-Qwen2.5-1.5B-instruct-Dataset](https://huggingface.co/datasets/Minsang/TSD-KD-Qwen2.5-1.5B-Instruct-Gen). We provide the trained checkpoint, [TSD-KD-Qwen2.5-1.5B](https://huggingface.co/Minsang/TSD-KD_Qwen2.5-1.5B).
Note: You only use the prompts during on-policy KD.
* For off-policy distillation, you can also use teacher-generated data [TSD-KD-Qwen2.5-14B-instruct](https://huggingface.co/datasets/Minsang/TSD-KD-Qwen2.5-14B-Instruct-Gen) and train using LAMBDA=0.0.

## TSD-KD Training Guide
1. Requirements Installation
Run this command to set up your environment:

```
pip install torch==2.5.1
pip install transformers==4.57.3
pip install trl==0.21.0
```

2. Training Script
You can copy and paste this entire block into your terminal or a .sh file. It includes the variables and the launch command.

```
# Hyperparameters
BETA=0.9
LAMBDA=1.0
THRESHOLD=0.1
MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
INDIRECT_KD_ALPHA=0.1

# Execute Training
accelerate launch --config_file accelerate_ddp_config.yaml train.py \
    --beta $BETA \
    --lambda $LAMBDA \
    --threshold $THRESHOLD \
    --model_name $MODEL_NAME \
    --indirect_kd_alpha $INDIRECT_KD_ALPHA
```


## Citation
If you use any part of this code and pretrained weights for your purpose, please cite our [paper](https://openreview.net/pdf?id=zph7e5JaXc).
```
@inproceedings{
    kim2026explain,
    title={Explain in Your Own Words: Improving Reasoning via Token-Selective Dual Knowledge Distillation},
    author={Minsang Kim and Seung Jun Baek},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=zph7e5JaXc}
}
```
