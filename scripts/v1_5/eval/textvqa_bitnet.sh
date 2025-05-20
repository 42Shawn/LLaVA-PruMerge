
#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-base ./bitnet_b1_58_3B \
    --model-path ./checkpoints/llava-bitnet-finetune-lora \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl  \
    --image-folder ./playground/data/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/llava-bit.jsonl \
    --temperature 0 \
    --conv-mode llama_2

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/llava-bit.jsonl

