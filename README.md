# TSD-KD
Explain in Your Own Words: Improving Reasoning via Token-Selective Dual Knowledge Distillation

#### Training from UltraInteract_sft [TSD-KD-Qwen2.5-1.5b-instruct](https://huggingface.co/datasets/Minsang/TSD-KD-Qwen2.5-1.5B-Instruct-Gen)
Note: You only use the prompts during on-policy KD.
* For off-policy distillation, you can also use teacher-generated data [TSD-KD-Qwen2.5-14b-instruct](https://huggingface.co/datasets/Minsang/TSD-KD-Qwen2.5-14B-Instruct-Gen) and train using LAMBDA=0.0.

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
@InProceedings{
  title = 	 {Explain in Your Own Words: Improving Reasoning via Token-Selective Dual Knowledge Distillation},
  author =       {Minsang Kim, Seungjun Baek},
  booktitle = 	 {The Fourteenth International Conference on Learning Representations (ICLR)},
  year = 	 {2026},
  series = 	 {Proceedings of ICLR},
  month = 	 {23--25 Apr},
  publisher =    {The Fourteenth International Conference on Learning Representations (ICLR)},
  pdf = 	 {https://openreview.net/pdf?id=zph7e5JaXc},
  abstract = 	 {Knowledge Distillation (KD) can transfer the reasoning abilities of large models to smaller ones, which can reduce the costs to generate Chain-of-Thoughts for reasoning tasks. KD methods typically ask the student to mimic the teacher's distribution over the entire output. However, a student with limited capacity can be overwhelmed by such extensive supervision causing a distribution mismatch, especially in complex reasoning tasks. We propose Token-Selective Dual Knowledge Distillation (TSD-KD), a framework for student-centric distillation. TSD-KD focuses on distilling important tokens for reasoning and encourages the student to explain reasoning in its own words. TSD-KD combines indirect and direct distillation. Indirect distillation uses a weak form of feedback based on preference ranking. The student proposes candidate responses generated on its own; the teacher re-ranks those candidates as indirect feedback without enforcing its entire distribution. Direct distillation uses distribution matching; however, it selectively distills tokens based on the relative confidence between teacher and student. Finally, we add entropy regularization to maintain the student's confidence during distillation. Overall, our method provides the student with targeted and indirect feedback to support its own reasoning process and to facilitate self-improvement. The experiments show the state-of-the-art performance of TSD-KD on 10 challenging reasoning benchmarks, outperforming the baseline and runner-up in accuracy by up to 54.4% and 40.3%, respectively. Notably, a student trained by TSD-KD even outperformed its own teacher model in four cases by up to 20.3%.}
  }
```
