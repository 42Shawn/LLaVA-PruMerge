
from bitnet_b1_58_3B.modeling_bitnet import BitnetForCausalLM
from bitnet_b1_58_3B.tokenization_bitnet import BitnetTokenizer

from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM

model, loading_info = BitnetForCausalLM.from_pretrained("/home/zl986/LLaVA-PruMerge-BitNet/bitnet_b1_58_3B",  output_loading_info=True)
tokenizer = BitnetTokenizer.from_pretrained("/home/zl986/LLaVA-PruMerge-BitNet/bitnet_b1_58_3B")

print(loading_info)

model = model.cuda()

prompt = "what do you think about math?"
inputs = tokenizer(prompt, return_tensors="pt")

generate_ids = model.generate(inputs.input_ids.cuda(), max_length=256, eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id)
print(generate_ids)
outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


print(outputs)
