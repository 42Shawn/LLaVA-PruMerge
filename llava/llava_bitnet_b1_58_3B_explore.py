import torch 
import sys

import json 
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer
from llava.model.language_model.llava_bitnet_b1_58_3B import LlavaBitnet_b1_58_3BConfig,LlavaBitnet_b1_58_3BForCausalLM
import llava.model.language_model.llava_bitnet_b1_58_3B as llava_olmo
import PIL
import torchvision 

device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)

with open('bitnet_b1_58_3B/config.json') as json_file:
    data = json.load(json_file)
config_class = llava_olmo.LlavaBitnet_b1_58_3BConfig(**data)

# config_class = LlavaBitnet_b1_58_3BConfig(**data)
model = LlavaBitnet_b1_58_3BForCausalLM(config_class).to(device)
model.load_state_dict(torch.load('Bitnet_b1_58_3B/pytorch_model.bin'))

model.eval()

# tokenizer = AutoTokenizer.from_pretrained("NousResearch/Bitnet_b1_58_3B")

tokenizer = AutoTokenizer.from_pretrained(
            "NousResearch/Bitnet_b1_58_3B",
            cache_dir="./cache/",
            model_max_length=1024,
            padding_side="right",
            pad_token_id=1,
            unk_token='<|padding|>',
            trust_remote_code=True
            ) 

text = "Paris is a historic city with architectural marvels. It is also "


inputs = tokenizer(text, return_tensors='pt', return_token_type_ids=False).to(device)

# response = model.generate(**inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)


# llava olmo setup 
image_tensor = torchvision.io.read_image('playground/data/LLaVA-Pretrain/images/00316/003163402.jpg')
lolmo = llava_olmo.LlavaOLMoBitnet1BForCausalLM(config_class).to(device)
lolmo.load_state_dict(torch.load('OLMo_Bitnet_1B/pytorch_model.bin'), strict=False)
response = lolmo.generate(inputs=inputs['input_ids'], max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)

print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])
