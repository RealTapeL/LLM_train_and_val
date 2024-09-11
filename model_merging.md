Model merging
模型合并
$ python scripts/merge_llama3_with_chinese_lora_low_mem.py \
    --base_model path_to_original_llama3_dir \
    --lora_model path_to_lora \
    --output_dir path_to_output_dir 