beta=0.9
lambda=1.0
threshold=0.1
model_name=Qwen/Qwen2.5-1.5B-Instruct
indirect_kd_alpha=0.1

accelerate launch --config_file accelerate_ddp_config.yaml train.py $beta $lambda $threshold $model_name $indirect_kd_alpha
