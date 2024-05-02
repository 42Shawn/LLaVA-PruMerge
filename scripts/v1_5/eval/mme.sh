# #!/bin/bash

#     --model-base lmsys/vicuna-7b-v1.5 \
#     --model-path /data/shangyuzhang/LLaVA/checkpoints/llava-v1.5-7b-lora-20240311-evit-token-prunemerge-advanced-8 \
    # --model-base lmsys/vicuna-7b-v1.5 \
    # --model-path /data/shangyuzhang/LLaVA/checkpoints/llava-v1.5-7b-lora-20240312-evit-token-prunemerge-advanced-adaptive \
    # --model-base lmsys/vicuna-7b-v1.5 \
    # --model-path /data/shangyuzhang/LLaVA/checkpoints/llava-v1.5-7b-lora-20240330-token-prunemerge-advanced_w_spacial_adaptive \

#     --model-path liuhaotian/llava-v1.5-7b \

    # --model-base lmsys/vicuna-13b-v1.5 \
    # --model-path /data/shangyuzhang/LLaVA/checkpoints/llava-v1.5-13b-lora-20240315-evit-token-prunemerge-advanced-8 \

python -m llava.eval.model_vqa_loader \
    --model-base lmsys/vicuna-7b-v1.5 \
    --model-path /data/shangyuzhang/LLaVA/checkpoints/llava-v1.5-7b-lora-20240311-evit-token-prunemerge-advanced-8 \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/llava-v1.5-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1


cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment llava-v1.5-7b

cd eval_tool

python calculation.py --results_dir answers/llava-v1.5-7b
