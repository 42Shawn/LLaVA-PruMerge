python -m llava.eval.model_vqa_science_efficient \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/efficient-llava-v1.5-7b-4bit-test.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/efficient-llava-v1.5-7b-4bit-test.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/efficient-llava-v1.5-7b-4bit-test_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/efficient-llava-v1.5-7b-4bit-test_result.json
