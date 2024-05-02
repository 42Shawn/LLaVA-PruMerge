#!/bin/bash XDG_CACHE_HOME='/data/shangyuzhang/'



python -m llava.eval.model_vqa_loader \
    --model-base lmsys/vicuna-7b-v1.5 \
    --model-path /data/shangyuzhang/LLaVA/checkpoints/llava-v1.5-7b-lora-to \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/llava-v1.5-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/llava-v1.5-7b.jsonl
