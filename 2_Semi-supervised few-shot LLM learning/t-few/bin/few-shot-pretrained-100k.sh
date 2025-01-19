#!/bin/bash
allow_skip_exp=True
eval_before_training=True
balanced_ibc=True

train_batch_size=2
grad_accum_factor=1

lr=0.003
re='^[0-9]+$'

cuda_device=0

# Set adaptively
num_steps=0
eval_epoch_interval=0

data_prob=65
# data_name=4

# for num_shot in 176
# 声明 num_shot 关联数组
declare -A num_shot_map
num_shot_map["2,0"]=128
num_shot_map["2,1"]=101
num_shot_map["2,32"]=51
num_shot_map["2,42"]=130
num_shot_map["2,1024"]=111
num_shot_map["4,0"]=145
num_shot_map["4,1"]=145
num_shot_map["4,32"]=76
num_shot_map["4,42"]=157
num_shot_map["4,1024"]=0


for model in 't03b'
# for model in 't011b' 't03b'
do
  # For zero-shot set to '0', for all to 'all'
  # for num_shot in 4
  for data_name in 2 4
  # for num_shot in 4 8 16 32 64 128
  # 
  # for num_shot in 4 8 16 32 64 128 256 512
  do
    # Datasets: car, income, heart, diabetes, jungle, bank, blood, calhousing, creditg, jungle
    # Run all serializations for car
    for dataset in heart 
    # for dataset in car car_list car_list_permuted car_list_shuffled car_list_values car_gpt car_t0 car_ttt
    do
      # for seed in 0
      for seed in 42 1024 0 1 32
      do
        # Zero-shot
        # eval_before_training=True
        # num_steps=0
        # Few-shot
        
        # 构造 key
        key="$data_name,$seed"       
        # 获取 num_shot
        num_shot=${num_shot_map[$key]}
        
        eval_before_training=False
        # num_steps=$(( 100 * ($num_shot/$train_batch_size)))
        num_steps=$((100 * num_shot / 2))  # 确保运算符和操作数都有效
        eval_epoch_interval=1

        CUDA_VISIBLE_DEVICES=${cuda_device} CONFIG_PATH=/root/t-few/configs HF_HOME=/root/.cache/huggingface \
        python -m src.pl_train -c ${model}.json+ia3.json+global.json -k dataset=${dataset} load_weight="pretrained_checkpoints/${model}_ia3_finish.pt" num_steps=${num_steps} num_shot=${num_shot} data_name=${data_name} data_prob=${data_prob}\
        exp_name=D1D2D3_${model}_${dataset}_numshot${data_name}_seed${seed}_prob${data_prob}_ia3_pretrained100k few_shot_random_seed=${seed} seed=${seed} allow_skip_exp=${allow_skip_exp} eval_before_training=${eval_before_training} eval_epoch_interval=${eval_epoch_interval} \
        batch_size=${train_batch_size} grad_accum_factor=${grad_accum_factor} lr=${lr}
      done
    done
  done
done


# for model in 't03b'
# # for model in 't011b' 't03b'
# do
#   # For zero-shot set to '0', for all to 'all'
#   # for num_shot in 4
#   for num_shot in 176
#   # for num_shot in 4 8 16 32 64 128
#   # 
#   # for num_shot in 4 8 16 32 64 128 256 512
#   do
#     # Datasets: car, income, heart, diabetes, jungle, bank, blood, calhousing, creditg, jungle
#     # Run all serializations for car
#     for dataset in heart 
#     # for dataset in car car_list car_list_permuted car_list_shuffled car_list_values car_gpt car_t0 car_ttt
#     do
#       # Zero-shot
#       # eval_before_training=True
#       # num_steps=0
#       # Few-shot
#       eval_before_training=False
#       num_steps=$(( 100 * ($num_shot / $train_batch_size)))
#       eval_epoch_interval=1

#       # for seed in 0
#       for seed in 42 1024 0 1 32
#       do
#         CUDA_VISIBLE_DEVICES=${cuda_device} CONFIG_PATH=/root/t-few/configs HF_HOME=/root/.cache/huggingface \
#         python -m src.pl_train -c ${model}.json+ia3.json+global.json -k dataset=${dataset} load_weight="pretrained_checkpoints/${model}_ia3_finish.pt" num_steps=${num_steps} num_shot=${num_shot} data_name=${data_name} data_prob=${data_prob}\
#         exp_name=D1D2D3_50_${model}_${dataset}_numshot${data_name}_seed${seed}_prob${data_prob}_ia3_pretrained100k few_shot_random_seed=${seed} seed=${seed} allow_skip_exp=${allow_skip_exp} eval_before_training=${eval_before_training} eval_epoch_interval=${eval_epoch_interval} \
#         batch_size=${train_batch_size} grad_accum_factor=${grad_accum_factor} lr=${lr}
#       done
#     done
#   done
# done
