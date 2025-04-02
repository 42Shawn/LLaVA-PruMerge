import os
import json
import argparse
import torch
import random
import glog

from lm_eval import evaluator
from eval_utils import LMEvalAdaptor
from .tokenization_bitnet import BitnetTokenizer
from .modeling_bitnet import BitnetForCausalLM


parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--hf_path', default='1bitLLM/bitnet_b1_58-3B', type=str)
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument("--tasks", type=str)
parser.add_argument("--output_path", default=None, type=str)
parser.add_argument('--num_fewshot', type=int, default=0)
parser.add_argument('--ctx_size', default=2048, type=int)


def main(args):
    model_str = args.hf_path
    model = BitnetForCausalLM.from_pretrained(
        args.hf_path,
        device_map='auto',
        low_cpu_mem_usage=True, 
        use_flash_attention_2=True,
        torch_dtype=torch.float16,
    ).half()

    tokenizer = BitnetTokenizer.from_pretrained(args.hf_path, use_fast=False)
    glog.info('loaded model!')

    task_names = args.tasks.split(",")

    lm_eval_model = LMEvalAdaptor(model_str, model, tokenizer, args.batch_size, args.ctx_size)
    results = evaluator.simple_evaluate(
        model=lm_eval_model,
        tasks=task_names,
        batch_size=args.batch_size,
        no_cache=True,
        num_fewshot=args.num_fewshot,
    )

    print(evaluator.make_table(results))

    if args.output_path is not None:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        # otherwise cannot save
        results["config"]["model"] = args.hf_path
        with open(args.output_path, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    main(args)
