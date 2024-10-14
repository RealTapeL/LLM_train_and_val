# 基于大语言模型的学科知识微调和验证
## Fine-tuning and validation of subject knowledge based on llm
### 此仓库用于对各类大语言模型的学科知识微调，包含了[指令精调数据](https://github.com/RealTapeL/Fine-tuning-and-validation-based-on-llama3/tree/main/data)、[模型效果评测](https://github.com/RealTapeL/Fine-tuning-and-validation-based-on-llama3/tree/main/scripts)以及[预训练](https://github.com/RealTapeL/Fine-tuning-and-validation-based-on-llama3/tree/main/scripts)和[使用有标注数据进行指令微调](https://github.com/RealTapeL/Fine-tuning-and-validation-based-on-llama3/blob/main/scripts/training/run_sft.sh)，支持[llama.cpp](https://github.com/ggerganov/llama.cpp)、[vLLM](https://github.com/vllm-project/vllm)等生态。

# 如何使用？
## How to use?🧐

***以下均在Linux系统测试，win与macos系统情况未知***
## 准备工作
确保您的机器有足够的显存加载完整模型以进行合并模型操作（建议显存>24G)
安装依赖库（项目根目录requirements.txt）
```
pip install -r requirements.txt
```


### TRAINING
你可以使用无标注数据进行模型的[预训练](https://github.com/RealTapeL/Fine-tuning-and-validation-of-subject-knowledge-based-on-llm/blob/main/scripts/training/Readme_pt.md)。
你还可以使用有标注数据进行[指令微调](https://github.com/RealTapeL/Fine-tuning-and-validation-of-subject-knowledge-based-on-llm/blob/main/scripts/training/Readme_sft.md)

### 模型合并
这一步骤会合并LoRA权重，生成全量模型权重（safetensors格式）。
```
$ python scripts/merge_llama3_with_chinese_lora_low_mem.py
--base_model path_to_original_llama3_dir
--lora_model path_to_lora
--output_dir path_to_output_dir
```
## 模型效果评测

### 模型生成效果评测
为了全面评估相关模型的性能，建议用户根据自身关注的任务特性，进行针对性的模型测试，以筛选出最适合特定任务需求的模型。[Fastchat Chatbot Arena](https://lmarena.ai/?arena)，推出了模型在线对战平台，可浏览和评测模型回复质量。对战平台提供了胜率、Elo评分等评测指标，并且可以查看两两模型的对战胜率等结果。

## 模型客观效果评测
### C-Eval
[C-Eval](https://github.com/hkust-nlp/ceval) 是一个全面的中文基础模型评估套件。它由 13948 道多项选择题组成，涵盖 52 个不同的学科和 4 个难度级别
在抱抱脸Hugging Face上下载评测数据集，并解压
```
wget https://huggingface.co/datasets/ceval/ceval-exam/resolve/main/ceval-exam.zip
unzip ceval-exam.zip -d data
```
### 运行预测脚本
```
cd scripts/ceval
python eval.py \
    --model_path ${model_path} \
    --few_shot False \
    --with_prompt False\
    --n_times 1 \
    --ntrain 5 \
    --do_save_csv False \
    --do_test False \
    --output_dir ${output_path} \
```

### 指令数据
以下是本项目的部分指令数据

 指令数据名称  | 说明  
 ----                                                                                                                              | ----- |
[stem_zh_instruction](https://github.com/RealTapeL/Fine-tuning-and-validation-based-on-llama3/tree/main/data/stem_zh_instruction)  | 使用gpt-3.5爬取的STEM数据，包含物理、化学、医学、生物学、地球科学 
[DPO-zh-en-emoji ](https://huggingface.co/datasets/shareAI/DPO-zh-en-emoji)                                                        | ShareAI是国内的优秀开源组织，他们整理的中文数据集可以对已有的对话模型进行语言载体和语言风格(幽默风格 + emoji 表情) 的偏好强化对齐

## [llama.cpp](https://github.com/ggerganov/llama.cpp)部署

运行前请确保环境：

系统应有make（MacOS/Linux自带）或cmake（Windows需自行安装）编译工具
建议使用Python 3.10以上编译和运行该工具

Step 1: 编译llama.cpp

llama.cpp在2024年4月30日对编译做出重大改动，请务必拉取最新仓库进行编译！
```
$ git clone https://github.com/ggerganov/llama.cpp
```
对llama.cpp项目进行编译，生成./main（用于推理）和./quantize（用于量化）二进制文件
```
$ make
```
Windows/Linux用户如需启用GPU推理，则推荐使用cuda编译
```
$ make LLAMA_CUDA=1
```

Step 2:生成量化版本(gguf格式)模型

目前llama.cpp已支持.safetensors文件以及Hugging Face格式.bin转换为FP16的GGUF格式
```
$ python convert_hf_to_gguf.py your_model_name
$ ./llama-quantize your_model_name/ggml-model-f16.gguf your_model_name/ggml-model-q4_0.gguf q4_0
```

Step 3: 加载并启动模型
## Linux、macOS等系统：
#### 对话模式（允许与模型持续交互）
```
./llama-cli -m /home/fw/results8.24/earthllm826/earth.gguf -p "You are a helpful assistant" -cnv --chat-template chatml
```
## Windows系统：
#### 对话模式（允许与模型持续交互）
```
./llama-cli.exe -m /home/fw/results8.24/earthllm826/earth.gguf -p "You are a helpful assistant" -cnv --chat-template chatml
```
