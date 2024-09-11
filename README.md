# 基于语言模型的学科知识微调和验证
## Fine-tuning and validation of subject knowledge based on llm
### 此仓库用于对各类语言模型的学科知识微调，包含了[指令精调数据](https://github.com/RealTapeL/Fine-tuning-and-validation-based-on-llama3/tree/main/data)、[模型效果评测](https://github.com/RealTapeL/Fine-tuning-and-validation-based-on-llama3/tree/main/scripts)以及[预训练](https://github.com/RealTapeL/Fine-tuning-and-validation-based-on-llama3/tree/main/scripts)和[使用有标注数据进行指令微调](https://github.com/RealTapeL/Fine-tuning-and-validation-based-on-llama3/blob/main/scripts/training/run_sft.sh)，支持[llama.cpp](https://github.com/ggerganov/llama.cpp), [vLLM](https://github.com/vllm-project/vllm)等生态。

# 如何使用？
## How to use?🧐

### TRAINING
你可以使用[预训练](https://github.com/RealTapeL/Fine-tuning-and-validation-based-on-llama3/tree/main/scripts)和[使用有标注数据进行指令微调](https://github.com/RealTapeL/Fine-tuning-and-validation-based-on-llama3/blob/main/scripts/training/run_sft.sh)进行模型的训练。
#### 预训练步骤
[预训练代码](https://github.com/RealTapeL/Fine-tuning-and-validation-based-on-llama3/blob/main/scripts/training/run_clm_pt_with_peft.py)
进入项目的`scripts/training`目录，运行`ash run_pt.sh`进行指令精调，默认使用单卡，***特别注意：运行前应先修改脚本并指定相关参数，脚本中的相关参数值仅供调试参考***。`run_pt.sh`内容为：
```
lr=1e-4
lora_rank=64
lora_alpha=128
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.05

pretrained_model=path/to/llm_model/dir
tokenizer_name_or_path=${pretrained_model}
dataset_dir=path/to/pt/data/dir
data_cache=temp_data_cache_dir
per_device_train_batch_size=1
gradient_accumulation_steps=8
block_size=1024
output_dir=output_dir

torchrun --nnodes 1 --nproc_per_node 1 run_clm_pt_with_peft.py \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${tokenizer_name_or_path} \
    --dataset_dir ${dataset_dir} \
    --data_cache_dir ${data_cache} \
    --validation_split_percentage 0.001 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --do_train \
    --low_cpu_mem_usage \
    --seed $RANDOM \
    --bf16 \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 3 \
    --save_steps 200 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --block_size ${block_size} \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --lora_dropout ${lora_dropout} \
    --modules_to_save ${modules_to_save} \
    --torch_dtype bfloat16 \
    --load_in_kbits 16 \
    --ddp_find_unused_parameters False
```
