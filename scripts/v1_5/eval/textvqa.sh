#!/bin/bash     --model-path liuhaotian/llava-v1.5-7b \     --model-base lmsys/vicuna-7b-v1.5 \ --model-path /data/shangyuzhang/LLaVA/checkpoints/llava-v1.5-7b-lora-20240222-4compresstoken \
#     --model-base lmsys/vicuna-7b-v1.5 \
#     --model-path /data/shangyuzhang/LLaVA/checkpoints/llava-v1.5-7b-lora-20240311-evit-token-prunemerge-advanced-8 \

#     --model-path liuhaotian/llava-v1.5-7b \

    # --model-base lmsys/vicuna-13b-v1.5 \
    # --model-path /data/shangyuzhang/LLaVA/checkpoints/llava-v1.5-13b-lora-20240315-evit-token-prunemerge-advanced-8 \

# model_name=$1
# model_name_replace=${model_name//\//_}
# echo $model_name_replace

python -m llava.eval.model_vqa_loader \
    --model-base lmsys/vicuna-7b-v1.5 \
    --model-path /data/shangyuzhang/LLaVA/checkpoints/llava-v1.5-7b-lora-20240312-evit-token-prunemerge-advanced-adaptive \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/llava-v1.5-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/llava-v1.5-7b.jsonl

# model_name=$1
# model_name_replace=${model_name//\//_}
# echo $model_name_replace

# python -m llava.eval.model_vqa_loader \
#     --model-path ./checkpoints/$model_name \
#     --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
#     --image-folder ./playground/data/eval/pope/val2014 \
#     --answers-file ./playground/data/eval/pope/answers/$model_name_replace.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# python llava/eval/eval_pope.py \
#     --annotation-dir ./playground/data/eval/pope/coco \
#     --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
#     --result-file ./playground/data/eval/pope/answers/$model_name_replace.jsonl