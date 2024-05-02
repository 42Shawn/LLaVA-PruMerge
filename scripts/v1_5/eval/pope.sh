#!/bin/bash XDG_CACHE_HOME='/data/shangyuzhang/'

#     --model-base lmsys/vicuna-7b-v1.5 \
#     --model-path /data/shangyuzhang/LLaVA/checkpoints/llava-v1.5-7b-lora-20240311-evit-token-prunemerge-advanced-8 \

#     --model-base lmsys/vicuna-7b-v1.5 \
#     --model-path /data/shangyuzhang/LLaVA/checkpoints/llava-v1.5-7b-lora-20240330-token-prunemerge-advanced_w_spacial_adaptive \

#     --model-path liuhaotian/llava-v1.5-7b \

    # --model-base lmsys/vicuna-13b-v1.5 \
    # --model-path /data/shangyuzhang/LLaVA/checkpoints/llava-v1.5-13b-lora-20240315-evit-token-prunemerge-advanced-8 \


python -m llava.eval.model_vqa_loader \
    --model-base lmsys/vicuna-7b-v1.5 \
    --model-path /data/shangyuzhang/LLaVA/checkpoints/llava-v1.5-7b-lora-20240312-evit-token-prunemerge-advanced-adaptive \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/llava-v1.5-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/llava-v1.5-7b.jsonl

# Category: random, # samples: 2910
# TP      FP      TN      FN
# 609     605     850     846
# Accuracy: 0.5013745704467354
# Precision: 0.5016474464579901
# Recall: 0.41855670103092785
# F1 score: 0.4563506931434994
# Yes ratio: 0.41718213058419246
# 0.456, 0.501, 0.502, 0.419, 0.417
# ====================================
# Category: adversarial, # samples: 3000
# TP      FP      TN      FN
# 1183    118     1382    317
# Accuracy: 0.855
# Precision: 0.9093005380476556
# Recall: 0.7886666666666666
# F1 score: 0.8446983220278472
# Yes ratio: 0.43366666666666664
# 0.845, 0.855, 0.909, 0.789, 0.434
# ====================================
# Category: popular, # samples: 3000
# TP      FP      TN      FN
# 1183    59      1441    317
# Accuracy: 0.8746666666666667
# Precision: 0.9524959742351047
# Recall: 0.7886666666666666
# F1 score: 0.8628738147337709
# Yes ratio: 0.414
# 0.863, 0.875, 0.952, 0.789, 0.414
# ====================================

# python -m llava.eval.model_vqa_loader \
#     --model-path liuhaotian/llava-v1.5-7b \
#     --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
#     --image-folder ./playground/data/eval/pope/val2014 \
#     --answers-file ./playground/data/eval/pope/answers/llava-v1.5-7b.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# python llava/eval/eval_pope.py \
#     --annotation-dir ./playground/data/eval/pope/coco \
#     --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
#     --result-file ./playground/data/eval/pope/answers/llava-v1.5-7b.jsonl

# Category: random, # samples: 2910
# TP      FP      TN      FN
# 614     604     851     841
# Accuracy: 0.5034364261168385
# Precision: 0.5041050903119869
# Recall: 0.4219931271477663
# F1 score: 0.45940890385334837
# Yes ratio: 0.41855670103092785
# 0.459, 0.503, 0.504, 0.422, 0.419
# ====================================
# Category: adversarial, # samples: 3000
# TP      FP      TN      FN
# 1187    133     1367    313
# Accuracy: 0.8513333333333334
# Precision: 0.8992424242424243
# Recall: 0.7913333333333333
# F1 score: 0.8418439716312057
# Yes ratio: 0.44
# 0.842, 0.851, 0.899, 0.791, 0.440
# ====================================
# Category: popular, # samples: 3000
# TP      FP      TN      FN
# 1187    69      1431    313
# Accuracy: 0.8726666666666667
# Precision: 0.9450636942675159
# Recall: 0.7913333333333333
# F1 score: 0.8613933236574746
# Yes ratio: 0.4186666666666667
# 0.861, 0.873, 0.945, 0.791, 0.419
# ====================================