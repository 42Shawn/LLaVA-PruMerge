#!/bin/bash


python -m llava.eval.model_vqa_science \
    --model-base lmsys/vicuna-13b-v1.5 \
    --model-path /data/shangyuzhang/LLaVA/checkpoints/llava-v1.5-13b-lora-20240315-to \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/llava-v1.5-7b_result.json

# python -m llava.eval.model_vqa_science \
#     --model-base lmsys/vicuna-7b-v1.5 \
#     --model-path /data/shangyuzhang/LLaVA/checkpoints/llava-v1.5-7b-lora-20240207 \
#     --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
#     --image-folder ./playground/data/eval/scienceqa/images/test \
#     --answers-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-lora.jsonl \
#     --single-pred-prompt \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# python llava/eval/eval_science_qa.py \
#     --base-dir ./playground/data/eval/scienceqa \
#     --result-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-lora.jsonl \
#     --output-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-lora_output.jsonl \
#     --output-result ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-lora_result.json